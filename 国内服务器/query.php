<?php
// === 配置区 ===
// 数据库配置（请替换为您的真实信息）
$pdo = new PDO('mysql:host=127.0.0.1;dbname=gogo_ai;charset=utf8mb4', 'root', 'your_password_here');
$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

// Kafka 配置（从 config.ini 读取或硬编码）
$kafka_brokers = '120.23.xx.xx:9092'; // 替换为您的 Kafka 地址
$topic = 'gogo-query';

// === 提交问题 ===
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'submit') {
    $task_id = uniqid('task_', true);
    $question = trim($_POST['question'] ?? '');

    if (empty($question)) {
        echo json_encode(['error' => '问题不能为空']);
        exit;
    }

    try {
        // 1. 写入 MySQL
        $stmt = $pdo->prepare("INSERT INTO query_results (task_id, status) VALUES (?, 'pending')");
        $stmt->execute([$task_id]);

        // 2. 发送到 Kafka
        $conf = new RdKafka\Conf();
        $conf->set('metadata.broker.list', $kafka_brokers);
        $producer = new RdKafka\Producer($conf);
        $kafka_topic = $producer->newTopic($topic);
        $kafka_topic->produce(RD_KAFKA_PARTITION_UA, 0, json_encode([
            'task_id' => $task_id,
            'question' => $question
        ]));
        $producer->flush(5000);

        echo json_encode(['status' => 'queued', 'task_id' => $task_id]);
    } catch (Exception $e) {
        echo json_encode(['error' => '系统错误']);
    }
    exit;
}

// === 轮询结果 ===
if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['action']) && $_GET['action'] === 'poll' && !empty($_GET['task_id'])) {
    $task_id = $_GET['task_id'];
    $stmt = $pdo->prepare("SELECT status, answer FROM query_results WHERE task_id = ?");
    $stmt->execute([$task_id]);
    $row = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($row) {
        echo json_encode([
            'status' => $row['status'],
            'answer' => $row['answer'] ?? ''
        ]);
    } else {
        echo json_encode(['status' => 'not_found']);
    }
    exit;
}

// === 默认响应 ===
echo json_encode(['error' => 'Invalid request']);
?>
