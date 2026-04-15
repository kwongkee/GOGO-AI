# GOGO-AI 架构文档

> **生成时间:** 2026-04-15  
> **技术栈:** ThinkPHP 5.0 / PHP 7.2  
> **服务端口:** 5000  
> **访问地址:** https://ai.gogo198.net  
> **数据来源:** 服务器真实代码扫描  

---

## 📊 项目概览

| 属性 | 值 |
|------|-----|
| **项目名称** | GOGO-AI（AI智能服务平台） |
| **技术框架** | ThinkPHP 5.0（非Python！） |
| **PHP版本** | 7.2 |
| **数据库** | MySQL |
| **部署方式** | Git直接部署 |
| **CI/CD** | GitHub Actions（智能6阶段流程） |
| **路由数量** | 与GOGO-Admin共享相同路由结构 |

---

## ⚠️ 重要说明

> **注意:** GitHub仓库名为 GOGO-AI，但服务器上运行的是 **ThinkPHP 5.0 PHP项目**，与 GOGO-Admin 使用相同的框架和路由结构！  
> 并非 Python AI 项目。项目目录与 GOGO-Admin 高度相似。

---

## 📁 目录结构

```
ai.gogo198.net/
│
├── 📂 application/                    # ThinkPHP应用核心 ⭐
│   ├── 📂 index/
│   │   ├── 📂 controller/             # 前台控制器（7个）
│   │   │   ├── Customer.php           # 客户管理
│   │   │   ├── Gather.php             # 数据采集
│   │   │   ├── Index.php              # 首页/信息展示
│   │   │   ├── Loggin.php             # 日志管理
│   │   │   ├── Member.php             # 会员管理
│   │   │   ├── Members.php            # 会员列表
│   │   │   └── Shop.php               # 店铺管理
│   ├── 📂 api/
│   │   └── 📂 controller/             # API控制器
│   ├── config.php                     # 应用配置
│   ├── common.php                     # 公共函数库
│   ├── route.php                      # 路由配置
│   └── database.php                   # 数据库配置
│
├── 📂 public/                         # Web根目录
│   └── index.php                      # 入口文件
├── 📂 thinkphp/                       # ThinkPHP框架核心
├── 📂 vendor/                         # Composer依赖
├── 📂 extend/                         # 自定义扩展
├── 📂 runtime/                        # 运行时缓存/日志
│   ├── cache/                         # 框架缓存
│   └── log/                           # 运行日志
│
├── 📄 think                           # 命令行工具
├── 📄 build.php                       # 构建脚本
├── 📄 composer.json                   # PHP依赖
├── 📄 composer.lock                   # 依赖版本锁定
└── 📄 MP_verify_*.txt                 # 微信域名验证
```

---

## 🎮 控制器详解

### 前台控制器 (`application/index/controller/`)

| 控制器 | 功能说明 | 备注 |
|--------|----------|------|
| **Index** | 首页/信息展示/客户背景调查 | 同Admin |
| **Member** | 个人会员管理 | 同Admin |
| **Members** | 会员列表查询 | 同Admin |
| **Customer** | 客户管理 | 同Admin |
| **Shop** | 店铺管理 | 同Admin |
| **Gather** | 数据采集管理 | 同Admin |
| **Loggin** | 系统日志管理 | 同Admin |

> **注意:** 相比 GOGO-Admin（11个控制器），GOGO-AI 少了 `Main.php`、`Memberc.php`、`Merchant.php`、`Monitor.php` 四个控制器，定位更精简。

---

## 🛤️ API路由文档（真实路由）

> 路由结构与 GOGO-Admin 高度一致

### 首页路由
| 方法 | 路由 | 控制器 | 说明 |
|------|------|--------|------|
| GET | `/` | `index/index/index` | 网站首页 |

### 信息展示模块（Index控制器）
| 方法 | 路由 | 说明 |
|------|------|------|
| ANY | `index/enterprise_news` | 购购动态 |
| ANY | `index/cross_news` | 跨境新闻 |
| ANY | `index/chooseMarket` | 选市场 |
| ANY | `index/customers` | 找客户 |
| ANY | `index/background_email` | 全球客户背景调查-邮箱 |
| ANY | `index/background_site` | 全球客户背景调查-网站 |
| ANY | `index/background_company` | 全球客户背景调查-企业 |
| ANY | `index/background_overseasreport` | 全球客户背景调查-信用报告 |
| ANY | `index/KYBreport` | KYB合规报告 |
| ANY | `index/searchengine` | 搜索引擎获客 |
| ANY | `index/domainsearch` | 域名获客 |
| ANY | `index/findcustomers` | 海关数据获客 |
| ANY | `index/enterprise` | 社交媒体获客 |

---

## 🔧 核心功能模块

### 1. 信息展示平台
- 跨境电商信息门户
- 企业动态与新闻
- 行业数据展示

### 2. 全球客户背景调查
- 企业信息查询
- 信用报告生成
- KYB合规检查
- 多渠道获客工具

### 3. 会员与店铺管理
- 会员注册与管理
- 店铺信息维护

### 4. 数据采集
- 网站数据采集
- 日志管理

---

## 🛠️ 部署信息

| 项目 | 值 |
|------|-----|
| **服务器** | 阿里云 ECS 39.108.11.214 (CentOS 7) |
| **部署路径** | `/www/wwwroot/ai.gogo198.net/` |
| **访问地址** | https://ai.gogo198.net |
| **服务端口** | 5000 |
| **运行用户** | www |
| **备份目录** | `/opt/backups/gogo-ai/` |

---

## 🔐 第三方集成

| 集成 | 说明 |
|------|------|
| 微信域名验证 | `MP_verify_UwFjMrSKelIbvktq.txt` |

---

## 📈 CI/CD 流程状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| 代码审核 | ✅ 已配置 | SonarQube + PHP语法检查 |
| 架构文档生成 | ✅ 已配置 | 自动生成docs/ARCHITECTURE.md |
| 修复建议 | ✅ 已配置 | GitHub Issue自动创建 |
| 部署 | ✅ 已配置 | SSH直接部署 |
| 邮件通知 | ✅ 已配置 | 发送至198@gogo198.net |

---

*由 GOGO CI/CD 基于服务器真实代码扫描生成 · 2026-04-15*
