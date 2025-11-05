<?php
require '../vendor/autoload.php';
use Predis\Client;

function compress_image_base64($base64_string, $quality=50) {
    error_log("压缩图片，质量: $quality");
    $img_data = base64_decode($base64_string);
    $img = imagecreatefromstring($img_data);
    if (!$img) {
        error_log("图片创建失败");
        return $base64_string; // 返回原图
    }
    
    ob_start();
    imagejpeg($img, null, $quality);
    $compressed_data = ob_get_clean();
    imagedestroy($img);
    
    return base64_encode($compressed_data);
}

class QueryProcessor {
    private $redis;
    private $config;
    private $log_file = '/var/log/multimodal_rag/query.log';
    
    // 数据库配置常量
    const DB_HOST = 'rm-wz9mt4j79jrdh0p3z.mysql.rds.aliyuncs.com';
    const DB_NAME = 'shop';
    const DB_USER = 'gogo198';
    const DB_PASS = 'Gogo@198';
    
    public function __construct() {
        $this->initializeLog();
        $this->initializeRedis();
        $this->loadConfig();
        $this->log("QueryProcessor初始化完成");
    }
    
    private function initializeLog() {
        // 确保日志目录存在
        $log_dir = dirname($this->log_file);
        if (!is_dir($log_dir)) {
            mkdir($log_dir, 0755, true);
        }
        $this->log("日志系统初始化完成");
    }
    
    private function log($message, $level = 'INFO') {
        $timestamp = date('Y-m-d H:i:s');
        $pid = getmypid();
        $log_message = "[$timestamp] [$level] [PID:$pid] $message\n";
        error_log($log_message, 3, $this->log_file);
    }
    
    private function initializeRedis() {
        try {
            $this->redis = new Client([
                'scheme' => 'tcp',
                'host'   => 'localhost',
                'port'   => 6379,
                'timeout' => 5.0,
            ]);
            $this->log("Redis连接成功");
        } catch (Exception $e) {
            $this->log("Redis连接失败: " . $e->getMessage(), 'ERROR');
            $this->sendErrorResponse("Redis服务不可用", 503);
        }
    }
    
    private function loadConfig() {
        $this->config = parse_ini_file('/app/config.ini');
        $this->config['max_file_size'] = $this->config['max_file_size'] ?? 100 * 1024 * 1024;
        $this->config['max_file_count'] = $this->config['max_file_count'] ?? 10;
        $this->config['allowed_extensions'] = ['mp3', 'mp4', 'jpg', 'png', 'jpeg', 'pdf', 'txt', 'pptx', 'docx', 'xls', 'xlsx'];
        $this->log("配置文件加载完成");
    }
    
    public function processQuery() {
        $this->log("开始处理查询请求");
        
        $question = $_POST['question'] ?? '';
        $this->log("用户问题: " . substr($question, 0, 100) . (strlen($question) > 100 ? '...' : ''));
        
        if(empty($question)){
            $this->log("问题内容为空", 'ERROR');
            $this->sendErrorResponse('消息内容不能为空！');
        }
        
        $images_base64 = $_POST['images_base64'] ?? [];
        $audios_base64 = $_POST['audios_base64'] ?? [];
        $videos_base64 = $_POST['videos_base64'] ?? [];
        $files = $_FILES ?? [];
        $mode = $_POST['mode'] ?? 'query';
        $chat_pid = isset($_POST['chat_pid'])?intval($_POST['chat_pid']):0;
        $chat_id = isset($_POST['chat_id'])?intval($_POST['chat_id']):0;
        $uid = isset($_POST['uid'])?intval($_POST['uid']):0;

        $this->log("请求参数 - mode: $mode, chat_pid: $chat_pid, chat_id: $chat_id, uid: $uid");
        $this->log("多媒体文件数量 - 图片: " . count($images_base64) . ", 音频: " . count($audios_base64) . ", 视频: " . count($videos_base64));

        // 处理查询模式 - 新增同步文件向量化流程
        return $this->processQueryModeWithSyncFileIndexing($question, $images_base64, $audios_base64, $videos_base64, $chat_pid, $uid, $chat_id);
    }
    
    /**
     * 获取数据库连接
     */
    private function getDBConnection() {
        try {
            $dsn = "mysql:host=" . self::DB_HOST . ";dbname=" . self::DB_NAME . ";charset=utf8mb4";
            $pdo = new PDO($dsn, self::DB_USER, self::DB_PASS);
            $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
            $pdo->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_ASSOC);
            return $pdo;
        } catch (PDOException $e) {
            $this->log("数据库连接失败: " . $e->getMessage(), 'ERROR');
            return false;
        }
    }
    
    private function safeJsonDecode($json_str) {
        $decoded = json_decode($json_str, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            $this->log("JSON解析失败: " . json_last_error_msg() . " | 原始: " . substr($json_str, 0, 500), 'ERROR');
            return null;
        }
        return $decoded;
    }
    
    /**
     * 统一的降级响应方法
     * 用于网络失败、解析失败、美国服务异常等情况
     */
    private function degradedResponse($reason = "服务暂时不可用",$chat_id,$chat_pid,$uid) {
        $this->log("触发降级响应: $reason", 'WARNING');
        // 插入一条 AI 回复记录（占位）
        $insert_id = $this->insertAiReply($uid,$chat_pid,$chat_id, "系统忙，请稍后重试");
        return [
            'status' => 'success',
            'insert_id' => $insert_id,
            'reply' => "系统忙，请稍后重试（原因：$reason）",
            'fallback' => true
        ];
    }
    
    /**
    * 插入 AI 回复到数据库（复用原有逻辑）
    */
    private function insertAiReply($uid,$chat_pid,$chat_id, $content) {
        try {
            $db = new \Think\Db();
            $id = $db->name('website_chatlist')->insertGetId([
                'uid' => $uid ?? 0,
                'chat_pid' => $chat_pid ?? 0,
                'content' => $content,
                'is_ai' => 1,
                'ai_times' => time(),
                'create_time' => date('Y-m-d H:i:s')
            ]);
            return $id;
        } catch (Exception $e) {
            $this->log("降级回复插入失败: " . $e->getMessage(), 'ERROR');
            return 0;
        }
    }
    
    private function compress_vectors($vectors) {
        $data = '';
        foreach ($vectors as $vec) {
            foreach ($vec as $v) $data .= pack('f', $v);
        }
        
        // 将 gzcompress 改为 gzencode 使用gzip格式
        $compressed = gzencode($data, 6);
        
        // 添加调试日志
        error_log("GZIP压缩前长度: " . strlen($data) . 
                  ", 压缩后长度: " . strlen($compressed) . 
                  ", 压缩头: " . bin2hex(substr($compressed, 0, 2)));
        
        return base64_encode($compressed);
    }
    
    private function safe_input($data) {
        return trim(htmlspecialchars($data, ENT_QUOTES, 'UTF-8'));
    }
    
    private function compress_payload($data) {
        return base64_encode(gzcompress(json_encode($data), 6));
    }
    
    /**
     * 同步文件向量化处理模式
     */
    private function processQueryModeWithSyncFileIndexing($question, $images_base64, $audios_base64, $videos_base64, $chat_pid, $uid, $chat_id) {
        $starttime = microtime(true);
        $this->log("进入查询模式，开始同步文件向量化流程");
        
        set_time_limit(300);
    
        // 1. 获取关联文件
        $this->log("获取关联文件，chat_id: $chat_id");
        $associated_files = $this->getAssociatedFiles($chat_id);
        
        $this->log("获取到关联文件数量: " . count($associated_files));
    
        $all_doc_vectors = [];
        $all_doc_contents = [];
    
        if (!empty($associated_files)) {
            $this->log("开始同步处理关联文件向量化");
    
            foreach ($associated_files as $file_info) {
                $this->log("处理文件: " . $file_info['name']);
                $file_result = $this->processFileSyncVectorization($file_info);
    
                if ($file_result['success']) {
                    $vector = $file_result['vector'] ?? [];
                    $content = $file_result['text_content'] ?? '';
    
                    if ($vector && $content) {
                        $all_doc_vectors[] = $vector;
                        $all_doc_contents[] = $content;
                    }
                    $this->log("文件向量化成功: {$file_info['name']}");
                } else {
                    $this->log("文件向量化失败: {$file_info['name']} - {$file_result['error']}", 'ERROR');
                }
            }
        }
    
        // ============================== 发送美国服务器 ==============================
        if (!empty($all_doc_vectors)) {
            $this->log("准备发送同步查询请求到美国服务器");
    
            $post_data = [
                'query' => $question,
                'doc_vectors' => $this->compress_vectors($all_doc_vectors),  // 768维向量
                'chat_id' => $chat_id,
                'sync_processing' => true,
                'metadata' => [  // 元数据替代，含checksum一致性
                    'vector_count' => count($all_doc_vectors),
                    'content_lengths' => array_map('strlen', $all_doc_contents),
                    'checksum' => md5(json_encode($all_doc_vectors))  // 加校验和
                ]
            ];
            
            // 安全加固
            $question = htmlspecialchars(trim($question), ENT_QUOTES, 'UTF-8');
            
            $question = $this->safe_input($question);
            $payload = $this->compress_payload([
                'q' => $question,
                'v' => $vectors,  // 768维
                'c' => $chat_id
            ]);
            
            $this->log("请求数据大小: " . strlen(json_encode($post_data)));
            
            $usa_url = 'http://47.254.122.81:5000/predict_sync';
            // $headers = ['Content-Type: application/json', 'X-API-Key: abc123321cba'];
            
            $this->log("优化后请求数据大小: " . strlen($json_data)); // 记录新大小
            
             // 增加超时时间到120秒
            $start_time = microtime(true);
            $response = $this->httpRequest($usa_url, $post_data, ['X-API-Key' => 'abc123321cba']);
            $request_time = round((microtime(true) - $start_time) * 1000, 2);
            
            $this->log("请求耗时: {$request_time}ms");
    
            $result = $this->safeJsonDecode($response);
            if ($result === null) {
                return $this->degradedResponse("美国服务器响应格式错误",$chat_id,$chat_pid,$uid);
            }
            
            if ($result['status'] !== 'success') {
                return $this->degradedResponse("美国服务器返回错误: " . ($result['error'] ?? '未知'));
            }
    
            // 解析响应
            if ($response !== false) {
                $result = json_decode($response, true);
                if ($result && $result['status'] === 'success') {
                    $this->log("美国服务器处理成功，返回答案");
                
                    // 直接返回美国服务器的结果
                    echo json_encode([
                        'status' => 'success',
                        'response' => $result['result'],
                        'source_documents' => $result['source_documents'] ?? [],
                        'sync_processed' => true,
                        'timestamp' => date('c')
                    ]);
                    exit;
                }else{
                    $this->log("美国服务器返回错误: " . ($result['error'] ?? '未知'));
                    // 降级 (匹配文档容错)
                    $this->log("进入普通查询模式降级方案");
                    // 已有降级...
                }
            }else{
                $this->log("请求失败，所有重试耗尽");
                // 降级
                $this->log("进入普通查询模式降级方案");
                // 已有降级...
            }
        }
    
        // 降级方案：如果没有向量数据或美国服务器失败，进入普通查询
        $this->log("进入普通查询模式降级方案");
        return $this->processNormalQuery($question, $chat_pid, $uid, $chat_id, $starttime);
    }
    
    private function httpRequestWithExtendedTimeout($url, $data, $headers = [], $timeout = 300) {
        $this->log("延长超时HTTP请求: $url, 超时: {$timeout}秒");
        
        $max_retries = 2;
        for ($attempt = 1; $attempt <= $max_retries; $attempt++) {
            $this->log("尝试第{$attempt}次请求，超时: {$timeout}秒");
            
            $ch = curl_init();
            curl_setopt($ch, CURLOPT_URL, $url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
            curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
            curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);
            curl_setopt($ch, CURLOPT_POST, 1);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $data);
            curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
            curl_setopt($ch, CURLOPT_TIMEOUT, $timeout); // 动态超时
            curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 15);
            
            $output = curl_exec($ch);
            $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            $error = curl_error($ch);
            curl_close($ch);
            
            if ($output !== false && $http_code === 200) {
                $this->log("延长超时请求成功: HTTP $http_code");
                return $output;
            } else {
                $this->log("延长超时请求失败: HTTP $http_code, 错误: $error");
                if ($attempt < $max_retries) {
                    sleep(5); // 失败后等待5秒再重试
                    continue;
                }
            }
        }
        return false;
    }
    
    // 2. 修改 processQueryModeWithSyncFileIndexing → 异步版
    private function processQueryModeWithAsyncIndexing($question, $chat_id) {
        $associated_files = $this->getAssociatedFiles($chat_id);
        
        if (empty($associated_files)) {
            return $this->sendErrorResponse('无关联文件');
        }
        
        $file_paths = array_map(function($f) {
            return $f['full_path']; // 确保是绝对路径
        }, $associated_files);
        
        // 异步启动向量化
        $task_id = $this->startVectorizeTask($file_paths, $chat_id);
        
        // 立即返回，让前端轮询
        echo json_encode([
            'status' => 'processing',
            'task_id' => $task_id,
            'message' => '文件处理中，请稍后提问',
            'poll_interval' => 3,
            'timestamp' => date('c')
        ]);
        exit;
    }
    
    private function processNormalQuery($question, $chat_pid, $uid, $chat_id, $starttime) {
        // 可选：走旧流程或直接返回空
        $this->sendErrorResponse("暂无知识文件支持该问题");
    }
    
    /**
     * 同步文件向量化处理
     */
    private function processFileSyncVectorization($file_info) {
        $base_path = '/www/wwwroot/boss.gogo198.cn/public';
        $relative_path = $file_info['path'];
        $full_path = realpath($base_path . $relative_path);
    
        if (!$full_path || !file_exists($full_path)) {
            $this->log("文件不存在: $full_path", 'ERROR');
            return ['success' => false, 'error' => '文件不存在'];
        }
    
        $file_size = filesize($full_path);
        if ($file_size > $this->config['max_file_size']) {
            return ['success' => false, 'error' => '文件过大'];
        }
    
        $this->log("修改文件状态: ".$file_info['task_id']);
        $task_id = $file_info['task_id'];
        $task_id = 'vec_' . substr(md5($file_path . microtime(true)), 0, 12);
    
        // ============================== $command 从这里构造 ==============================
        $escaped_path = escapeshellarg($full_path);
        $command = "cd /app && " .
                   "source /app/rag_multi_env/bin/activate && " .
                   "python vectorize.py sync_process ". escapeshellarg(json_encode([$full_path, $task_id])) ." 2>&1";
        // =============================================================================
    
        $this->log("开始同步向量化文件: {$file_info['name']}");
        $this->log("执行命令: $command");
    
        $python_output = shell_exec($command . " 2>&1");
        // 修复：更鲁棒的JSON提取
        $json_str = $this->extractJsonFromOutput($python_output);
        $output_json = json_decode($json_str, true);
        
        if (json_last_error() !== JSON_ERROR_NONE || !isset($result['success'])) {
            $this->log("JSON解析失败，尝试降级解析", 'WARNING');
            // 降级方案：尝试从最后一行提取JSON
            $lines = explode("\n", trim($python_output));
            $last_line = end($lines);
            $result = json_decode($last_line, true);
        }
        
        if (json_last_error() === JSON_ERROR_NONE && isset($output_json['success']) && $output_json['success']) {
            $this->log("文件向量化成功: " . $file['name']);
            $success_count++;
        } else {
            $error_msg = $output_json['error'] ?? $python_output;
            $this->log("文件向量化失败: " . $file['name'] . " - " . $error_msg, 'ERROR');
        }
    
        // ============================== 鲁棒 JSON 解析（第2点） ==============================
        $json_start = strrpos($python_output, '{');
        if ($json_start === false) {
            $this->log("Python输出中未找到JSON起始符 '{'", 'ERROR');
            $this->log("完整输出: " . substr($python_output, -500), 'ERROR');
            return ['success' => false, 'error' => '无JSON输出'];
        }
    
        $json_str = substr($python_output, $json_start);
        $result = json_decode($json_str, true);
    
        if (!$result || !isset($result['success'])) {
            $this->log("JSON解析失败或格式错误", 'ERROR');
            $this->log("尝试解析的JSON: " . substr($json_str, 0, 200), 'ERROR');
            return ['success' => false, 'error' => 'JSON解析失败'];
        }
    
        if (!$result['success']) {
            $error = $result['error'] ?? '未知错误';
            $this->log("Python向量化失败: $error", 'ERROR');
            return ['success' => false, 'error' => $error];
        }
        // =============================================================================
    
        $this->log("文件向量化成功: {$file_info['name']}, 向量维度: " . count($result['vector']));
    
        return [
            'success' => true,
            'vector' => $result['vector'],
            'text_content' => $result['text_content'] ?? '',
            'text_length' => $result['text_length'] ?? 0
        ];
    }
    
    private function extractJsonFromOutput($output) {
        // 方法1：查找最后一个完整的JSON对象
        $pattern = '/\{(?:[^{}]|(?R))*\}/';
        preg_match_all($pattern, $output, $matches);
        
        if (!empty($matches[0])) {
            // 使用最后一个匹配的JSON（最可能是最终结果）
            $last_match = end($matches[0]);
            if (json_decode($last_match) !== null) {
                return $last_match;
            }
        }
        
        // 方法2：提取第一行和最后一行之间的内容
        $lines = explode("\n", trim($output));
        $candidate_lines = [];
        
        foreach ($lines as $line) {
            $line = trim($line);
            if (strpos($line, '{') === 0 && strpos($line, '}') === strlen($line)-1) {
                $candidate_lines[] = $line;
            }
        }
        
        if (!empty($candidate_lines)) {
            return end($candidate_lines);
        }
        
        // 方法3：降级方案 - 返回原始输出
        return $output;
    }
    
     /**
     * 调用Python进行同步向量化
     */
    private function callPythonVectorizationSync($file_path, $task_id) {
        $this->log("开始同步向量化文件: $file_path");
    
        $args = json_encode([$file_path]);
        $escaped = escapeshellarg($args);
        $cmd = "cd /app && "
             . "source /app/rag_multi_env/bin/activate && "
             . "python vectorize.py sync_process $escaped 2>&1";
        
        $this->log("执行命令: $cmd");
        $output = shell_exec($cmd);
        
        $this->log("Python输出完整长度: " . strlen($output));
        
        // 改进的结果解析逻辑
        $lines = explode("\n", $output);
        $json_output = '';
        
        // 查找JSON输出
        foreach ($lines as $line) {
            if (strpos($line, '{') === 0 || strpos($line, '{"success"') !== false) {
                $json_output = $line;
                break;
            }
        }
        
        // 如果没找到，使用最后一行
        if (empty($json_output)) {
            $json_output = end($lines);
        }
        
        $this->log("提取的JSON输出: " . substr($json_output, 0, 200));
        
        $result = json_decode($json_output, true);
        if ($result && isset($result['success']) && $result['success']) {
            // 发送 Kafka 消息（使用修正后的方法）
            $kafka_result = $this->sendVectorToKafka($result, $file_path, $task_id);
            if ($kafka_result) {
                return [
                    'success' => true,
                    'vector_dimension' => $result['vector_dimension'],
                    'text_length' => $result['text_length']
                ];
            } else {
                return ['success' => false, 'error' => 'Kafka消息发送失败'];
            }
        }
        
        return ['success' => false, 'error' => $output ?: 'Python脚本执行失败'];
    }
    
    private function sendVectorToKafka($vector_data, $file_path, $task_id) {
        try {
            $this->log("尝试发送向量数据到Kafka: $task_id");
            
            $message = [
                'id' => $task_id,
                'type' => 'multimodal_document', 
                'path' => $file_path,
                'vector' => $vector_data['vector'],
                'content' => $vector_data['text_content'],
                'metadata' => [
                    'source' => 'sync_query',
                    'chat_id' => $this->current_chat_id ?? 0,
                    'timestamp' => date('c'),
                    'content_length' => strlen($vector_data['text_content']),
                    'vector_dimension' => count($vector_data['vector'])
                ]
            ];
            
            $message_for_checksum = $message;
            $check_data_str = json_encode($message_for_checksum, JSON_UNESCAPED_UNICODE);
            $message['checksum'] = md5($check_data_str);
            
            $success = $this->sendToKafkaBroker($message);
            
            if ($success) {
                $this->log("向量数据发送到Kafka成功: $task_id");
                return true;
            } else {
                $this->log("向量数据发送到Kafka失败: $task_id - 但不阻塞流程", 'WARNING');
                return false; // 返回false但不抛出异常
            }
            
        } catch (Exception $e) {
            $this->log("发送向量数据异常（非阻塞）: " . $e->getMessage(), 'WARNING');
            return false;
        }
    }
    
    /**
     * 发送消息到Kafka代理
     */
    private function sendToKafkaBroker($message) {
        $max_retries = 3;
        $retry_delay = 2; // 秒
        
        for ($attempt = 1; $attempt <= $max_retries; $attempt++) {
            try {
                $this->log("开始发送Kafka消息 (尝试 {$attempt}/{$max_retries}): " . $message['id']);
                
                $kafka_rest_url = 'http://39.108.11.214:9092/topics/vector-data';
                
                $payload = [
                    'records' => [
                        [
                            'value' => $message
                        ]
                    ]
                ];
                
                $ch = curl_init($kafka_rest_url);
                curl_setopt($ch, CURLOPT_POST, 1);
                curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload, JSON_UNESCAPED_UNICODE));
                curl_setopt($ch, CURLOPT_HTTPHEADER, [
                    'Content-Type: application/vnd.kafka.json.v2+json',
                    'Accept: application/vnd.kafka.v2+json'
                ]);
                curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
                curl_setopt($ch, CURLOPT_TIMEOUT, 30);
                curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 15); // 连接超时
                
                $response = curl_exec($ch);
                $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
                $curl_error = curl_error($ch);
                curl_close($ch);
                
                if ($http_code === 200) {
                    $this->log("Kafka消息发送成功: " . $message['id']);
                    return true;
                } else {
                    $this->log("Kafka消息发送失败，HTTP代码: {$http_code}, 响应: {$response}, 错误: {$curl_error}", 'WARNING');
                    
                    if ($attempt < $max_retries) {
                        sleep($retry_delay);
                        continue;
                    }
                    return false;
                }
                
            } catch (Exception $e) {
                $this->log("Kafka发送异常 (尝试 {$attempt}/{$max_retries}): " . $e->getMessage(), 'WARNING');
                
                if ($attempt < $max_retries) {
                    sleep($retry_delay);
                    continue;
                }
                return false;
            }
        }
    }
    
    /**
     * 等待美国服务器索引更新
     */
    private function waitForUSIndexUpdate($wait_seconds = 10) {
        $this->log("等待美国服务器索引更新，等待时间: {$wait_seconds}秒", 'info');
        sleep($wait_seconds); // 简单等待，实际中可以更复杂的检查机制
        $this->log("美国服务器索引更新等待完成", 'info');
    }
    
    /**
     * 获取关联文件信息
     */
    private function getAssociatedFiles($chat_id) {
        if ($chat_id <= 0) {
            $this->log("chat_id无效: $chat_id", 'INFO');
            return [];
        }
        
        $pdo = $this->getDBConnection();
        if (!$pdo) {
            $this->log("数据库连接失败，无法获取关联文件", 'info');
            return [];
        }
        
        try {
            $sql = 'SELECT associations FROM ims_website_chatlist WHERE id = :chat_id';
            $stmt = $pdo->prepare($sql);
            $stmt->execute([':chat_id' => $chat_id]);
            $result = $stmt->fetch();
            
            if (!$result) {
                $this->log("未找到chat_id对应的记录: $chat_id", 'INFO');
                return [];
            }
            
            if (empty($result['associations'])) {
                $this->log("associations字段为空", 'INFO');
                return [];
            }
            
            $associations = json_decode($result['associations'], true);
            if (json_last_error() !== JSON_ERROR_NONE) {
                $this->log("associations JSON解析失败: " . json_last_error_msg(), 'ERROR');
                return [];
            }
            
            $files = $associations['files'] ?? [];
            $this->log("成功获取关联文件，数量: " . count($files));
            
            return $files;
            
        } catch (Exception $e) {
            $this->log("获取关联文件失败：" . $e->getMessage(), 'ERROR');
            return [];
        }
    }
    
    /**
     * 处理单个文件进行向量化索引
     */
    private function processSingleFileForIndexing($file_info) {
        $file_path = $file_info['path'] ?? '';
        $file_name = $file_info['name'] ?? '';
        
        if (empty($file_path)) {
            return null;
        }
        
        // 转换为绝对路径
        $absolute_path = $this->convertToAbsolutePath($file_path);
        if (!file_exists($absolute_path)) {
            $this->log("文件不存在: {$absolute_path}", 'info');
            return null;
        }
        
        // 验证文件类型和大小
        $file_extension = strtolower(pathinfo($absolute_path, PATHINFO_EXTENSION));
        if (!in_array($file_extension, $this->config['allowed_extensions'])) {
            $this->log("不支持的文件类型: {$file_extension}", 'info');
            return null;
        }
        
        $file_size = filesize($absolute_path);
        if ($file_size > $this->config['max_file_size']) {
            $this->log("文件过大: {$file_name}, 大小: {$file_size}", 'info');
            return null;
        }
        
        try {
            // 创建文件向量化任务
            $task_id = uniqid('file_index_task_', true);
            
            $task_data = [
                'task_id' => $task_id,
                'file_paths' => [$absolute_path],
                'original_path' => $file_path,
                'metadata' => [
                    'source' => 'user_query_associated',
                    'file_name' => $file_name,
                    'file_size' => $file_size,
                    'file_extension' => $file_extension,
                    'process_time' => date('c')
                ],
                'action' => 'process_existing_file'
            ];
            
            // 推送到Redis队列
            $this->redis->rpush('index_tasks', json_encode($task_data));
            
            // 初始化任务状态
            $this->redis->hset("task:{$task_id}", "status", "pending");
            $this->redis->hset("task:{$task_id}", "file_path", $absolute_path);
            $this->redis->hset("task:{$task_id}", "original_path", $file_path);
            $this->redis->hset("task:{$task_id}", "create_time", date('c'));
            $this->redis->hset("task:{$task_id}", "file_name", $file_name);
            $this->redis->expire("task:{$task_id}", 86400);
            
            // 启动Celery任务
            $celery_task_id = $this->startCeleryTask($task_id, [$absolute_path]);
            if ($celery_task_id) {
                $this->redis->hset("task:{$task_id}", "celery_task_id", $celery_task_id);
            }
            
            $this->log("文件向量化任务创建成功: {$file_name}, 任务ID: {$task_id}", 'info');
            return $task_id;
            
        } catch (Exception $e) {
            $this->log("创建文件向量化任务失败: " . $e->getMessage(), 'info');
            return null;
        }
    }
    
    /**
     * 等待文件处理完成
     */
    private function waitForFileProcessing($task_ids, $timeout_seconds = 180) {
        $start_time = time();
        $completed_tasks = [];
        
        while ((time() - $start_time) < $timeout_seconds) {
            $all_completed = true;
            
            foreach ($task_ids as $task_id) {
                if (in_array($task_id, $completed_tasks)) {
                    continue;
                }
                
                $task_status = $this->getTaskStatus($task_id);
                if ($task_status['success'] && $task_status['status'] === 'completed') {
                    $completed_tasks[] = $task_id;
                    $this->log("文件向量化任务完成: {$task_id}", 'info');
                } elseif ($task_status['success'] && $task_status['status'] === 'failed') {
                    $completed_tasks[] = $task_id;
                    $this->log("文件向量化任务失败: {$task_id}", 'info');
                } else {
                    $all_completed = false;
                }
            }
            
            if ($all_completed || count($completed_tasks) === count($task_ids)) {
                break;
            }
            
            sleep(2); // 等待2秒再次检查
        }
        
        return count($completed_tasks) === count($task_ids);
    }
    
    /**
     * 转换相对路径为绝对路径
     */
    private function convertToAbsolutePath($file_path) {
        // 如果已经是绝对路径，直接返回
        // if (strpos($file_path, '/') === 0) {
        //     return $file_path;
        // }
        
        // 相对路径转换为绝对路径
        $base_path = '/www/wwwroot/boss.gogo198.cn/public';
        $absolute_path = $base_path . '/' . ltrim($file_path, '/');
        
        // 检查文件是否存在
        if (file_exists($absolute_path)) {
            return $absolute_path;
        }
        
        // 如果不存在，尝试另一个可能的基础路径
        $base_path2 = '/www/wwwroot/website.gogo198.net/public';
        $absolute_path2 = $base_path2 . '/' . ltrim($file_path, '/');
        
        if (file_exists($absolute_path2)) {
            return $absolute_path2;
        }
        
        // 如果都不存在，返回原始路径（让后续处理报错）
        return $file_path;
    }
    
    private function processLearnMode($files) {
        if (count($files) > $this->config['max_file_count']) {
            $this->sendErrorResponse("文件数量过多，最大：{$this->config['max_file_count']}", 413);
        }
        
        $filePaths = [];
        foreach ($files as $file) {
            $ext = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
            if (!in_array($ext, $this->config['allowed_extensions'])) {
                $this->sendErrorResponse("不支持的文件类型：$ext", 415);
            }
            if ($file['size'] > $this->config['max_file_size']) {
                $max_size_mb = $this->config['max_file_size'] / 1024 / 1024;
                $this->sendErrorResponse("文件过大，最大：{$max_size_mb}MB", 413);
            }
            
            // 创建上传目录
            $upload_dir = "/uploads/knowledge_files/" . date('Ymd');
            if (!is_dir($upload_dir)) {
                mkdir($upload_dir, 0755, true);
            }
            
            $filePath = $upload_dir . '/' . basename($file['name']);
            if (!move_uploaded_file($file['tmp_name'], $filePath)) {
                $this->sendErrorResponse("文件保存失败：{$file['name']}", 500);
            }
            $filePaths[] = $filePath;
        }
        
        // 创建向量化任务
        $task_id = uniqid('learn_task_', true);
        $task_data = [
            'task_id' => $task_id,
            'file_paths' => $filePaths,
            'metadata' => [
                'source' => 'user_upload_query',
                'process_time' => date('c'),
                'mode' => 'learn'
            ],
            'action' => 'process_existing_file'
        ];
        
        // 推送到Redis队列
        $this->redis->rpush('index_tasks', json_encode($task_data));
        
        // 初始化任务状态
        $this->redis->hset("task:{$task_id}", "status", "pending");
        $this->redis->hset("task:{$task_id}", "file_paths", json_encode($filePaths));
        $this->redis->hset("task:{$task_id}", "create_time", date('c'));
        $this->redis->hset("task:{$task_id}", "mode", "learn");
        $this->redis->expire("task:{$task_id}", 86400);
        
        // 启动Celery任务
        $celery_task_id = $this->startCeleryTask($task_id, $filePaths);
        if ($celery_task_id) {
            $this->redis->hset("task:{$task_id}", "celery_task_id", $celery_task_id);
        }
        
        http_response_code(202);
        echo json_encode([
            "status" => "processing", 
            "task_id" => $task_id,
            "celery_task_id" => $celery_task_id,
            "message" => "文件已加入处理队列，正在向量化..."
        ]);
        exit;
    }
    
    private function processQueryMode($question, $images_base64, $audios_base64, $videos_base64, $chat_pid, $uid) {
        $starttime = microtime(true);
        
        // 设置更长的脚本执行时间
        set_time_limit(180);
        
        // 压缩图片
        foreach ($images_base64 as &$img) {
            if (strlen($img) > 10 * 1024 * 1024) {
                $img = compress_image_base64($img, 50);
            }
        }
        unset($img);
        
        // 准备查询数据
        $query_data = [
            'text' => $question,
            'images' => $images_base64,
            'audios' => $audios_base64,
            'videos' => $videos_base64
        ];
        
        // 发送到美国服务器的查询服务
        $ch = curl_init('http://47.254.122.81:5000/predict');
        curl_setopt($ch, CURLOPT_POST, 1);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($query_data));
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_TIMEOUT, 120);
        
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        curl_close($ch);
        
        if ($http_code !== 200) {
            $this->sendErrorResponse("远程服务响应错误: HTTP {$http_code}");
        }
        
        $result = json_decode($response, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            $this->sendErrorResponse("响应数据解析失败");
        }
        
        // 如果有聊天上下文，保存到数据库
        if ($chat_pid > 0 && isset($result['content'])) {
            $this->saveToDatabase($result, $chat_pid, $uid, $starttime);
        } else {
            echo json_encode([
                'status' => 'success',
                'response' => $result,
                'timestamp' => date('c')
            ]);
        }
    }
    
    private function startCeleryTask($task_id, $file_paths) {
        try {
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
            
            $this->log("构建的参数JSON: " . $args_json, 'info');
            
            $command = "cd /app && " .
                      "source /app/rag_multi_env/bin/activate && " .
                      "celery -A vectorize call vectorize.process_file " .
                      "--args={$escaped_args} --queue=file_processing";  // 使用正确的变量名
            
            $output = shell_exec($command . " 2>&1");
            
            if (preg_match('/"id": "([a-f0-9-]+)"/', $output, $matches)) {
                return $matches[1];
            }
            
            $this->log("Celery任务启动输出: " . $output, 'info');
            return null;
            
        } catch (Exception $e) {
            $this->log("启动Celery任务失败: " . $e->getMessage(), 'info');
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
                'file_paths' => json_decode($task_data['file_paths'] ?? '[]', true),
                'create_time' => $task_data['create_time'] ?? '',
                'update_time' => $task_data['update_time'] ?? '',
                'error' => $task_data['error'] ?? '',
                'celery_task_id' => $task_data['celery_task_id'] ?? '',
                'mode' => $task_data['mode'] ?? 'unknown'
            ];
            
        } catch (Exception $e) {
            return ['success' => false, 'error' => $e->getMessage()];
        }
    }
    
    private function httpRequest($url, $data, $headers = []) {
        $max_retries = 3;  // 匹配文档重试
        $retry_delay = 3;
        $this->log("HTTP请求开始: $url");
        $this->log("请求数据大小: " . strlen(json_encode($data)));
        
        for ($attempt = 1; $attempt <= $max_retries; $attempt++) {
            $this->log("尝试第{$attempt}次请求");
            
            $ch = curl_init($url);
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POST, true);
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
            curl_setopt($ch, CURLOPT_HTTPHEADER, array_merge($headers, ['Content-Type: application/json']));
            curl_setopt($ch, CURLOPT_TIMEOUT, 500);  // 500s，匹配文档service_timeout_threshold
            curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 15);
            curl_setopt($ch, CURLOPT_ENCODING, 'gzip');  // 压缩
            curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, true);  // 安全
            curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, 2);
            
            $response = curl_exec($ch);
            $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            $error = curl_error($ch);
            curl_close($ch);
            
            if ($response !== false && $http_code === 200) {
                $this->log("请求成功: HTTP $http_code");
                return $response;
            } else {
                $this->log("请求失败: HTTP $http_code, 错误: $error");
                if ($attempt < $max_retries) {
                    sleep($retry_delay);
                }
            }
        }
        $this->log("所有重试失败");
        return false;  // 上层降级
    }

    private function saveToDatabase($result, $chat_pid, $uid, $starttime) {
        // 修复：确保 $processed_files_count 有默认值
        $processed_files_count = $processed_files_count ?? 0;
        
        $this->log("开始保存到数据库，回复内容长度: " . strlen($result['content']));
        $this->log("回复内容预览: " . substr($result['content'], 0, 200));
        
        // 验证回答内容
        if (empty($result['content']) || strlen(trim($result['content'])) < 10) {
            $this->log("回答内容过短或为空，不保存到数据库", 'WARNING');
            echo json_encode([
                'status' => 'success',
                'insert_id' => 0,
                'response' => $result['content'] ?? '回答生成中，请稍后查看...',
                'timestamp' => date('c')
            ]);
            return;
        }
        
        function removeThinkTags($input) {
            $pattern = '/<think>.*?<\/think>\s*/s';
            return preg_replace($pattern, '', $input);
        }

        try {
            $pdo = $this->getDBConnection();
            if (!$pdo) {
                $this->sendErrorResponse('数据库连接失败');
            }
    
            #1、查找上级聊天记录
            $sql = 'SELECT * FROM ims_website_chatlist WHERE id = :chat_pid';
            $stmt = $pdo->prepare($sql);
            $stmt->execute([':chat_pid' => $chat_pid]);
            $pinfo = $stmt->fetch();
    
            if (!$pinfo) {
                $this->sendErrorResponse('上级聊天记录不存在');
            }
    
            // 将答案作为回复，隐藏<think>推理
            $cleanText = $this->removeThinkTags($result['content']);
    
            #2、记录AI回答
            $insert_sql = 'INSERT INTO ims_website_chatlist 
                      (company_id, pid, uid, who_send, is_read, content_type, `language`, origin_content, content, ip, is_ai, ai_times, createtime) 
                      VALUES 
                      (:company_id, :pid, :uid, :who_send, :is_read, :content_type, :language2, :origin_content, :content, :ip, :is_ai, :ai_times, :createtime)';
    
            $stmt2 = $pdo->prepare($insert_sql);
    
            $endtime = microtime(true);
            $executiontime = $endtime - $starttime;
    
            $cleanText = $this->removeThinkTags($result['content']);
            
            $params = [
                ':company_id' => $pinfo['company_id'],
                ':pid' => $pinfo['id'],
                ':uid' => $pinfo['uid'],
                ':who_send' => 1,
                ':is_read' => 0,
                ':content_type' => 1,
                ':language2' => $pinfo['language'],
                ':origin_content' => json_encode($cleanText, JSON_UNESCAPED_UNICODE),
                ':content' => '', // 翻译后填入
                ':ip' => $_SERVER['REMOTE_ADDR'],
                ':is_ai' => 1,
                ':ai_times' => number_format($executiontime, 2),
                ':createtime' => time()
            ];
    
            $stmt2->execute($params);
            $insertId = $pdo->lastInsertId();
    
            if ($insertId > 0) {
                $this->log("✅ 回答成功保存到数据库，ID: $insertId");
                
                #3、按照用户的语种，翻译回用户的语种，并保留当前国地语种的回答
                $this->httpRequest('https://shop.gogo198.cn/collect_website/public/?s=api/getgoods/translate_answer',
                    ['answer_id'=>$insertId, 'is_convert_customer'=>1, 'translate_back_language'=>$pinfo['language']]);
                
                echo json_encode([
                    'status' => 'success',
                    'insert_id' => $insertId,
                    'response' => $cleanText,
                    'source_documents' => $result['source_documents'] ?? [],
                    'sync_processing' => true,
                    'timestamp' => date('c')
                ]);
            } else {
                $this->log("❌ 数据库保存失败，使用直接返回");
                
                echo json_encode([
                    'status'=>'success', 
                    'insert_id' => 0, 
                    'chat_id' => $pinfo['id'],
                    'response' => $cleanText,
                    'files_processed' => $processed_files_count > 0,
                    'processed_count' => $processed_files_count,
                    'sync_processing' => true,
                    'timestamp' => date('c')
                ]);
            }
        } catch (PDOException $e) {
            $this->log("数据库错误: " . $e->getMessage(), 'ERROR');
            $this->sendErrorResponse('数据库操作失败，请稍后重试：'.$e->getMessage());
        } catch (Exception $e) {
            $this->log("业务错误: " . $e->getMessage(), 'ERROR');
            $this->sendErrorResponse($e->getMessage());
        }
    }
    
    private function sendErrorResponse($message, $status_code = 400) {
        http_response_code($status_code);
        echo json_encode([
            'status' => 'error',
            'message' => $message,
            'status_code' => $status_code,
            'timestamp' => date('c')
        ]);
        exit;
    }
    
    private function startVectorizeTask($file_paths, $chat_id) {
        $task_id = uniqid('vec_', true);
        
        $args = [
            $file_paths,
            $task_id,
            $chat_id
        ];
        
        // 记录任务状态
        $this->redis->hset("task:{$task_id}", "status", "pending");
        $this->redis->hset("task:{$_id}", "chat_id", $chat_id);
        $this->redis->hset("task:{$task_id}", "create_time", date('c'));
        $this->redis->expire("task:{$task_id}", 3600); // 1小时
        
        $command = "cd /app && source /app/rag_multi_env/bin/activate && " .
                   "celery -A vectorize call vectorize.process_file " .
                   "--args='" . json_encode($args) . "' --queue=file_processing";
        
        // 异步执行，不等待
        $output = shell_exec($command . " > /dev/null 2>&1 & echo $!");
        $celery_pid = trim($output);
        
        return $task_id;
    }
    
    // 4. 新增：最终查询美国服务器（不发向量）
    private function queryUSServer($question, $chat_id) {
        $url = "http://47.254.122.81:5000/predict";
        $data = [
            'query' => $question,
            'chat_id' => $chat_id
        ];
        
        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json', 'X-API-Key: abc123321cba']);
        curl_setopt($ch, CURLOPT_TIMEOUT, 15); // 15秒超时
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $curl_error = curl_error($ch);
        curl_close($ch);
        
        if ($http_code === 200) {
            return json_decode($response, true);
        } else {
            return ['status' => 'error', 'message' => '美国服务器超时'];
        }
    }
}

// 处理请求
// 3. 新增：轮询接口 /query_status?task_id=xxx
if (isset($_GET['action']) && $_GET['action'] === 'status' && isset($_GET['task_id'])) {
    $task_id = $_GET['task_id'];
    $status = $this->redis->hget("task:{$task_id}", "status");
    
    if ($status === 'completed') {
        // 触发最终查询
        $result = $this->queryUSServer($question, $chat_id);
        echo json_encode($result);
    } else {
        echo json_encode(['status' => $status ?? 'pending']);
    }
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $context = stream_context_create([
        'http' => [
            'timeout' => 120,  // 120秒超时
            'ignore_errors' => true
        ]
    ]);
    
    try {
        // 检查是否是状态查询请求
        if (isset($_GET['action']) && $_GET['action'] === 'status' && isset($_GET['task_id'])) {
            $processor = new QueryProcessor();
            $status = $processor->getTaskStatus($_GET['task_id']);
            echo json_encode($status);
            exit;
        }
        
        $processor = new QueryProcessor();
        $processor->processQuery();
    } catch (Exception $e) {
        $this->log("处理器异常: " . $e->getMessage(), 'info');
        http_response_code(500);
        echo json_encode([
            'status' => 'error',
            'message' => '处理器异常: ' . $e->getMessage(),
            'timestamp' => date('c')
        ]);
    }
} else {
    http_response_code(405);
    echo json_encode([
        'status' => 'error',
        'message' => '只支持POST请求',
        'timestamp' => date('c')
    ]);
}
?>