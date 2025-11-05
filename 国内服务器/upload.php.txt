<?php
// 引入 Predis 自动加载
require '../vendor/autoload.php';
use Predis\Client;

error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);
ini_set('error_log', '/var/log/multimodal_rag/file_upload.log');

header('Content-Type: application/json; charset=utf-8');

class FilePathProcessor {
    private $redis;
    private $allowed_extensions;
    private $celery_broker = 'redis://localhost:6379/0';
    
    public function __construct() {
        $this->initializeRedis();
        $this->loadConfig();
    }
    
    private function initializeRedis() {
        try {
            $this->redis = new Client([
                'scheme' => 'tcp',
                'host'   => '127.0.0.1',
                'port'   => 6379,
                'timeout' => 5.0,
            ]);
            
            $this->redis->ping();
            
        } catch (Exception $e) {
            error_log("Redis连接失败: " . $e->getMessage());
            $this->sendErrorResponse("Redis服务不可用", 503);
        }
    }
    
    private function loadConfig() {
        $this->allowed_extensions = ['mp3', 'mp4', 'jpg', 'png', 'jpeg', 'pdf', 'txt', 'pptx', 'docx', 'xls', 'xlsx'];
    }
    
    public function processFilePath() {
        try {
            if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
                $this->sendErrorResponse('只支持POST请求', 405);
            }
            
            // 获取参数
            $file_path = $_POST['file_path'] ? json_decode($_POST['file_path'],true): '';
            $is_user = $_POST['is_user'] ?? 0;
            $knowledge_id = $_POST['knowledge_id'] ?? 0;
            $cid = $_POST['cid'] ?? 0;
            
            if (empty($file_path)) {
                $this->sendErrorResponse('文件路径不能为空', 400);
            }
            
            // 处理文件路径（支持数组和字符串）
            $file_paths = is_array($file_path) ? $file_path : [$file_path];
            $results = [];
            
            foreach ($file_paths as $path) {
                $result = $this->processSingleFilePath($path, [
                    'is_user' => $is_user,
                    'knowledge_id' => $knowledge_id,
                    'cid' => $cid
                ]);
                $results[] = $result;
            }
            
            $success_count = count(array_filter($results, function($r) { 
                return $r['success']; 
            }));
            
            $this->sendSuccessResponse([
                'success_count' => $success_count,
                'total_files' => count($file_paths),
                'results' => $results
            ]);
            
        } catch (Exception $e) {
            error_log("文件路径处理异常: " . $e->getMessage());
            $this->sendErrorResponse('服务器内部错误', 500);
        }
    }
    
    private function processSingleFilePath($file_path, $metadata = []) {
        // 验证文件路径
        if (empty($file_path)) {
            return ['success' => false, 'error' => '文件路径为空'];
        }
        
        // 标准化路径
        if($metadata['cid']==0){
            //普通用户体验上传
            $full_path = '/www/wwwroot/ai.gogo198.net/public' . $file_path;
        }
        elseif($metadata['cid']>0){
            //商家上传知识文件
            $full_path = '/www/wwwroot/website.gogo198.net/public' . $file_path;
        }
        
        // 验证文件存在性
        if (!file_exists($full_path)) {
            return ['success' => false, 'error' => "文件不存在: {$file_path}"];
        }
        
        if (!is_readable($full_path)) {
            return ['success' => false, 'error' => "文件不可读: {$file_path}"];
        }
        
        // 验证文件扩展名
        $file_extension = strtolower(pathinfo($full_path, PATHINFO_EXTENSION));
        if (!in_array($file_extension, $this->allowed_extensions)) {
            return ['success' => false, 'error' => "不支持的文件类型: {$file_extension}"];
        }
        
        // 验证文件大小
        $file_size = filesize($full_path);
        $max_size = 100 * 1024 * 1024; // 100MB
        if ($file_size > $max_size) {
            $max_size_mb = $max_size / 1024 / 1024;
            return ['success' => false, 'error' => "文件过大，最大支持{$max_size_mb}MB"];
        }
        
        try {
            // 创建任务
            $task_id = uniqid('path_task_', true);
            
            $task_data = [
                'task_id' => $task_id,
                'file_paths' => [$full_path],
                'original_path' => $file_path,
                'metadata' => array_merge($metadata, [
                    'source' => 'backend_sync',
                    'file_size' => $file_size,
                    'file_extension' => $file_extension,
                    'process_time' => date('c')
                ]),
                'action' => 'process_existing_file'
            ];
            
            // 推送到Redis队列 - 使用 Predis 方法
            $this->redis->rpush('index_tasks', json_encode($task_data));
            
            // 初始化任务状态
            $this->redis->hset("task:{$task_id}", "status", "pending");
            $this->redis->hset("task:{$task_id}", "file_path", $full_path);
            $this->redis->hset("task:{$task_id}", "create_time", date('c'));
            $this->redis->hset("task:{$task_id}", "original_path", $file_path);
            $this->redis->expire("task:{$task_id}", 86400); // 24小时过期
            
            // 启动Celery任务（异步执行）
            $celery_task_id = $this->startCeleryTask($task_id, [$full_path]);
            if ($celery_task_id) {
                $this->redis->hset("task:{$task_id}", "celery_task_id", $celery_task_id);
            }
            
            return [
                'success' => true,
                'task_id' => $task_id,
                'celery_task_id' => $celery_task_id,
                'file_path' => $file_path,
                'file_size' => $file_size,
                'message' => '文件已加入处理队列'
            ];
            
        } catch (Exception $e) {
            return [
                'success' => false,
                'error' => '队列处理失败: ' . $e->getMessage()
            ];
        }
    }
    
    private function startCeleryTask($task_id, $file_paths) {
        try {
            // 使用shell执行Celery任务
            $escaped_paths = escapeshellarg(json_encode($file_paths));
            $escaped_task_id = escapeshellarg($task_id);
            
            // 构建参数数组
            $args_array = [
                $file_paths,  // 文件路径数组
                $task_id      // 任务ID
            ];
            
            // 转换为 JSON 并转义
            $args_json = json_encode($args_array, JSON_UNESCAPED_SLASHES);
            $escaped_args = escapeshellarg($args_json);  // 修复：正确的变量名
            
            error_log("构建的参数JSON: " . $args_json);
            
            $command = "cd /app && " .
                      "source /app/rag_multi_env/bin/activate && " .
                      "celery -A vectorize call vectorize.process_file " .
                      "--args={$escaped_args} --queue=file_processing";  // 使用正确的变量名
                   
            
            $output = shell_exec($command . " 2>&1");
            
            // 解析Celery任务ID
            if (preg_match('/"id": "([a-f0-9-]+)"/', $output, $matches)) {
                return $matches[1];
            }
            
            error_log("Celery任务启动输出: " . $output);
            return null;
            
        } catch (Exception $e) {
            error_log("启动Celery任务失败: " . $e->getMessage());
            return null;
        }
    }
    
    public function getTaskStatus($task_id) {
        try {
            if (!$this->redis->exists("task:{$task_id}")) {
                return ['success' => false, 'error' => '任务不存在'];
            }
            
            $task_data = $this->redis->hgetall("task:{$task_id}");
            return [
                'success' => true,
                'task_id' => $task_id,
                'status' => $task_data['status'] ?? 'unknown',
                'file_path' => $task_data['file_path'] ?? '',
                'original_path' => $task_data['original_path'] ?? '',
                'create_time' => $task_data['create_time'] ?? '',
                'update_time' => $task_data['update_time'] ?? '',
                'error' => $task_data['error'] ?? '',
                'celery_task_id' => $task_data['celery_task_id'] ?? ''
            ];
            
        } catch (Exception $e) {
            return ['success' => false, 'error' => $e->getMessage()];
        }
    }
    
    private function sendSuccessResponse($data) {
        http_response_code(200);
        echo json_encode([
            'success' => true,
            'data' => $data,
            'timestamp' => date('c')
        ], JSON_UNESCAPED_UNICODE);
        exit;
    }
    
    private function sendErrorResponse($message, $status_code = 400) {
        http_response_code($status_code);
        echo json_encode([
            'success' => false,
            'error' => $message,
            'status_code' => $status_code,
            'timestamp' => date('c')
        ], JSON_UNESCAPED_UNICODE);
        exit;
    }
}

// 处理请求
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        // 检查是否是状态查询请求
        if (isset($_GET['action']) && $_GET['action'] === 'status' && isset($_GET['task_id'])) {
            $processor = new FilePathProcessor();
            $status = $processor->getTaskStatus($_GET['task_id']);
            echo json_encode($status);
            exit;
        }
        
        $processor = new FilePathProcessor();
        $processor->processFilePath();
    } catch (Exception $e) {
        http_response_code(500);
        echo json_encode([
            'success' => false, 
            'error' => '处理器异常: ' . $e->getMessage(),
            'timestamp' => date('c')
        ]);
    }
} else {
    http_response_code(405);
    echo json_encode([
        'success' => false,
        'error' => '只支持POST请求',
        'timestamp' => date('c')
    ]);
}
?>