# 项目提示词管理和 Trace 数据上报分析

> 分析时间：2026年1月8日  
> 项目：zxy-workflow

## 概述

本项目使用 **CozeLoop** 平台实现了：
1. **提示词的中心化管理**（Prompt Hub）
2. **分布式追踪的自动上报**（Trace/Observability）

通过与 LangChain 和 Apollo 配置中心的深度集成，实现了提示词的版本管理、灰度发布和全链路监控。

---

## 一、提示词管理

### 1.1 架构设计

项目通过 **CozeLoop Prompt Hub** 进行提示词的中心化管理，而不是将提示词硬编码在代码中。

#### 核心组件

**配置文件**: `app/config.py`
```python
# CozeLoop 配置
cozeloop_workspace_id: str      # 工作空间 ID
cozeloop_api_token: str         # API 令牌
cozeloop_prompt_label: str      # 提示词版本标签（production/gray）
```

**提示词常量定义**: `app/core/config/constants.py`
```python
class CozePromptHubKey(Enum):
    # 主工作流
    MAIN_INTENT_CLASSIFY_PROMPT = "main_intent_classify_prompt"
    MAIN_SUMMARY_PROMPT = "main_summary_prompt"
    
    # 跨境ins
    ABROAD_INS_THINK_PROMPT = 'abroad_ins_think_prompt'
    ABROAD_INS_MAIN_PARAM_PARSE_PROMPT = "abroad_ins_main_param_parse_prompt"
    ABROAD_INS_SORT_TYPE_PARSE_PROMPT = "abroad_ins_sort_type_parse_prompt"

    # 知款ins
    ZHIKUAN_INS_THINK_PROMPT = "zhikuan_ins_think_prompt"
    ZHIKUAN_INS_MAIN_PARAM_PARSE_PROMPT = "zhikuan_ins_main_parse_prompt"
    ZHIKUAN_INS_SORT_TYPE_PARSE_PROMPT = "zhikuan_ins_sort_type_parse_prompt"

    # 小红书
    ZXH_XHS_THINK_PROMPT = 'zxh_xhs_think_prompt'
    ZXH_XHS_MAIN_PARAM_PARSE_PROMPT = 'zxh_xhs_main_param_parse_prompt'
    ZXH_XHS_SORT_TYPE_PARSE_PROMPT = 'zxh_xhs_sort_type_parse_prompt'
    ZXH_XHS_TOPIC_RAG_CLEAN_PROMPT = 'zxh_xhs_topic_rag_clean_prompt'

    # 抖衣
    DOUYI_THINK_PROMPT = 'douyi_think_prompt'
    DOUYI_MAIN_PARAM_PARSE_PROMPT = 'douyi_main_param_parse_prompt'
    DOUYI_SORT_TYPE_PARSE_PROMPT = 'douyi_sort_type_parse_prompt'
    DOUYI_USER_TAG_PARSE_PROMPT = 'douyi_user_tag_parse_prompt'
    DOUYI_PROPERTY_WRAP_PROMPT = 'douyi_property_wrap_prompt'
    DOUYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'douyi_category_vector_clean_prompt'
    DOUYI_PROPERTY_CLEAN_PROMPT = "douyi_property_clean_prompt"

    # 知衣
    ZHIYI_THINK_PROMPT = 'zhiyi_think_prompt'
    ZHIYI_MAIN_PARAM_PARSE_PROMPT = 'zhiyi_main_param_parse_prompt'
    ZHIYI_SORT_TYPE_PARSE_PROMPT = 'zhiyi_sort_type_parse_prompt'
    ZHIYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'zhiyi_category_vector_clean_prompt'
    ZHIYI_KB_FILTER_PROMPT = 'zhiyi_kb_filter_prompt'
    ZHIYI_USER_TAG_PARSE_PROMPT = 'zhiyi_user_tag_parse_prompt'
    ZHIYI_PROPERTY_CLEAN_PROMPT = "zhiyi_property_clean_prompt"
    ZHIYI_SHOP_CLEAN_PROMPT = "zhiyi_shop_clean_prompt"

    # 海外探款商品
    ABROAD_GOODS_THINK_PROMPT = 'abroad_goods_think_prompt'
    ABROAD_GOODS_MAIN_PARAM_PARSE_PROMPT = 'abroad_goods_main_param_parse_prompt'
    ABROAD_GOODS_SORT_TYPE_PARSE_PROMPT = 'abroad_goods_sort_type_parse_prompt'
    ABROAD_GOODS_STYLE_PARSE_PROMPT = 'abroad_goods_style_parse_prompt'
    ABROAD_GOODS_SITE_CLEAN_PROMPT = 'abroad_goods_site_clean_prompt'
    ABROAD_GOODS_BRAND_CLEAN_PROMPT = 'abroad_goods_brand_clean_prompt'

    # 图片设计
    IMAGE_PROMPT_OPTIMIZE = 'image_prompt_optimize'
```

**客户端封装**: `app/core/clients/coze_loop_client.py`

### 1.2 提示词获取和使用流程

#### API 接口

```python
class CozeLoopClientProvider:
    def get_prompt(self, prompt_key: str, label: Optional[str] = None) -> Prompt:
        """获取 CozeLoop 提示词"""
        if label is None:
            label = settings.cozeloop_prompt_label
        return self.get_client().get_prompt(prompt_key=prompt_key, label=label)

    def format_prompt(self, prompt: Prompt, variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用 CozeLoop 内置方法格式化提示词"""
        return self.get_client().prompt_format(prompt, variables)

    def get_langchain_messages(
        self,
        prompt_key: str,
        variables: Dict[str, Any],
        label: Optional[str] = None,
    ) -> List[BaseMessage]:
        """获取格式化后的 LangChain 消息列表（一站式方法）"""
        # 1. 获取提示词
        prompt = self.get_prompt(prompt_key, label=label)
        # 2. 使用 CozeLoop 格式化
        formatted = self.format_prompt(prompt, variables)
        # 3. 转换为 LangChain 消息
        return self._convert_to_langchain_messages(formatted)
```

#### 工作流程

1. **从 CozeLoop 远程获取提示词模板**（按 key + label）
2. **使用 CozeLoop 的 `prompt_format` 方法填充变量**
3. **自动转换为 LangChain Messages 格式**（`SystemMessage`/`HumanMessage`/`AIMessage`）

#### 使用示例

```python
# 文件: app/service/chains/workflow/select/zhiyi_graph.py
def _generate_thinking_and_report_task() -> None:
    invoke_params = {
        "user_query": format_query,
        "preferred_entity": req.preferred_entity,
        "industry": req.industry.split("#")[0] if req.industry else "",
        "user_preferences": req.user_preferences,
        "now_time": datetime.now().strftime("%Y-%m-%d"),
    }

    # 从 CozeLoop 获取提示词并填充变量
    messages = coze_loop_client_provider.get_langchain_messages(
        prompt_key=CozePromptHubKey.ZHIYI_THINK_PROMPT.value,
        variables=invoke_params,
    )
    
    # 获取 LLM 并调用
    llm = llm_factory.get_llm(
        LlmProvider.OPENROUTER.name, 
        LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
    )
    thinking_text = llm.invoke(messages)
```

### 1.3 灰度发布和版本管理

#### 版本标签

- **production**: 生产环境版本
- **gray**: 灰度环境版本

#### 动态切换

通过 Apollo 配置中心的 `cozeloop_prompt_label` 配置项动态切换版本：
- 无需修改代码
- 无需重启服务
- 支持实时切换

#### 降级机制

```python
# 文件: app/service/chains/workflow/main_orchestrator_graph.py
try:
    messages = coze_loop_client_provider.get_langchain_messages(
        prompt_key=CozePromptHubKey.MAIN_INTENT_CLASSIFY_PROMPT.value,
        variables=prompt_params,
    )
except Exception as e:
    logger.warning(f"[意图分类] 无法从 CozeLoop 获取提示词，使用内置提示词: {e}")
    messages = self._build_fallback_intent_prompt(prompt_params)
```

如果 CozeLoop 获取失败，会自动使用代码中的内置提示词作为降级方案。

### 1.4 提示词管理的优势

✅ **中心化管理**：多人协作时避免冲突，统一维护  
✅ **版本控制**：支持灰度发布，可快速迭代优化  
✅ **变量化设计**：提高提示词的复用性  
✅ **可追溯**：与 trace 系统集成，可查看每次调用使用的提示词版本  
✅ **动态更新**：修改提示词后，无需重新部署服务  

---

## 二、Trace 数据上报

### 2.1 架构设计

项目使用 **CozeLoop** 作为统一的可观测性平台，通过集成 LangChain 的 Callback 机制实现自动追踪。

#### 核心组件

- **LoopTracer**: LangChain 回调处理器，自动捕获 LLM 调用、链执行、工具调用等
- **Span**: 代表一个执行单元（工作流、节点、LLM 调用等）

### 2.2 Trace 上报流程

#### 工作流级别的 Trace

```python
# 文件: app/service/chains/workflow/base_graph.py
def run(self, req: WorkflowRequest) -> dict[str, Any]:
    client = coze_loop_client_provider.get_client()
    
    # 1. 创建根 Span
    with client.start_span(self.span_name, "workflow") as root_span:
        # 2. 设置上下文元数据
        root_span.set_deployment_env(settings.environment)
        root_span.set_user_id_baggage(str(req.user_id))
        root_span.set_message_id_baggage(req.message_id)
        root_span.set_thread_id_baggage(req.session_id)
        root_span.set_input(req)
        root_span.set_service_name(self.span_name)
        
        # 3. 创建 LangChain 回调处理器
        cozeloop_callback_handler = coze_loop_client_provider.create_trace_callback_handler(
            modify_name_fn=name_modifier  # 自定义 span 名称
        )
        
        # 4. 执行工作流（自动记录所有 LangChain 操作）
        state = self._compiled_graph.invoke(
            {"request": req},
            RunnableConfig(callbacks=[cozeloop_callback_handler])
        )
        
        return state
```

#### 回调处理器创建

```python
# 文件: app/core/clients/coze_loop_client.py
def create_trace_callback_handler(self, modify_name_fn=None, add_tags_fn=None):
    """创建独立的 callback handler"""
    return LoopTracer.get_callback_handler(
        self.get_client(), 
        modify_name_fn=modify_name_fn,  # 修改 span 名称的函数
        add_tags_fn=add_tags_fn          # 添加 span tags 的函数
    )
```

### 2.3 Trace 数据层次结构

```
根 Span: workflow_graph (手动创建)
│
├── 子 Span: 意图分类 (LangChain 自动捕获)
│   ├── LLM 调用
│   │   ├── 输入: messages
│   │   ├── 输出: intent_result
│   │   ├── 耗时: 1.2s
│   │   └── Token: input=150, output=50
│   └── 结构化输出解析
│
├── 子 Span: 思维链生成
│   └── LLM 调用
│       ├── 输入: user_query, industry, ...
│       ├── 输出: thinking_text
│       └── 耗时: 2.5s
│
├── 子 Span: 知识库检索
│   ├── 向量检索
│   │   ├── 查询: "时尚风格"
│   │   └── 结果: 5条
│   └── LLM 清洗
│       └── LLM 调用
│
└── 子 Span: 参数解析
    ├── 并行 LLM 调用 1: 排序类型
    ├── 并行 LLM 调用 2: 用户标签
    └── 并行 LLM 调用 3: 店铺清洗
```

### 2.4 上报的数据内容

#### 基础信息
- **用户 ID**: `user_id`
- **会话 ID**: `session_id` (thread_id)
- **消息 ID**: `message_id`
- **环境标识**: `environment` (gray/prod)
- **服务名称**: `service_name`

#### 执行数据
- **输入参数**: 请求对象、变量等
- **输出结果**: 响应数据、中间结果
- **执行耗时**: 每个 span 的时长
- **状态码**: 成功/失败状态

#### LLM 指标
- **模型名称**: 使用的具体模型
- **Token 消耗**: input tokens, output tokens
- **模型参数**: temperature, max_tokens 等
- **提示词版本**: 使用的提示词 key 和 label

#### 错误信息
- **异常堆栈**: 完整的错误堆栈
- **错误类型**: 异常类名
- **错误上下文**: 发生时的 state 数据

#### 自定义标签
通过 `add_tags_fn` 可添加业务标签，如：
- 工作流类型
- 行业分类
- 查询类型
- 数据源等

### 2.5 Span 名称自定义

为了让 trace 更易读，项目支持通过回调函数修改 span 名称：

```python
# 文件: app/service/chains/workflow/select/zhiyi_graph.py
def _get_trace_name_modifier(self):
    """返回 span 名称修改函数"""
    def name_modifier(node_name: str) -> str:
        name_map = {
            "init_state": "知衣选品",
            "_think_node": "生成思维链",
            "_query_selection_node": "查询品类维表",
            "_main_param_parse_node": "解析主要参数",
            "_sort_type_parse_node": "解析排序类型",
            "_db_query_node": "数据库查询",
            "_property_clean_node": "属性清洗",
            "_shop_clean_node": "店铺清洗",
            "_format_response_node": "格式化响应",
        }
        return name_map.get(node_name, node_name)
    return name_modifier
```

### 2.6 异常追踪

当发生错误时，自动记录异常信息并上报：

```python
try:
    # 执行工作流
    state = self._compiled_graph.invoke(...)
except AppException as e:
    root_span.set_error(e)  # 标记 span 状态为 error
    logger.exception(f"工作流执行失败 [trace_id={trace_id}]")
    raise
except Exception as e:
    root_span.set_error(e)
    raise AppException(...) from e
```

### 2.7 分布式追踪

通过 `session_id` 和 `message_id` 实现分布式追踪：

```python
trace_id = f"{req.session_id}_{req.message_id}"
logger.info(f"工作流执行成功 [trace_id={trace_id}]")
```

同一个会话的多轮对话可以通过 `session_id` 关联，同一条消息的多个服务调用可以通过 `message_id` 关联。

### 2.8 Trace 上报的优势

✅ **自动捕获**：集成 LangChain Callback，无需手动埋点  
✅ **完整链路**：记录从请求到响应的全部过程  
✅ **实时监控**：实时查看 LLM 性能和成本  
✅ **分布式追踪**：支持跨服务、跨工作流的追踪  
✅ **错误定位**：快速定位问题节点和错误原因  
✅ **性能优化**：识别性能瓶颈，优化慢节点  

---

## 三、配合 Apollo 配置中心

### 3.1 配置动态管理

项目使用 Apollo 配置中心管理 CozeLoop 相关配置：

```python
# 文件: app/config.py
class Settings:
    _DEFAULTS: Dict[str, Any] = {
        # CozeLoop 配置
        "cozeloop_workspace_id": None,
        "cozeloop_api_token": None,
        "cozeloop_prompt_label": "gray",  # 默认使用灰度版本
    }
```

### 3.2 环境隔离

通过不同的 Apollo Meta Server 实现环境隔离：

- **GRAY**: `http://192.168.200.37:8022`
- **PROD**: `http://192.168.200.37:8021`

### 3.3 热更新

Apollo 配置支持热更新，修改配置后无需重启服务：

```python
# 注册配置热更新回调
apollo_provider = settings.get_apollo_provider()
apollo_provider.on_config_change("cozeloop_prompt_label", callback_fn)
```

---

## 四、最佳实践

### 4.1 提示词命名规范

```
<工作流名称>_<功能模块>_<操作类型>_prompt

示例:
- zhiyi_think_prompt              # 知衣-思考-提示词
- abroad_goods_main_param_parse_prompt  # 海外探款-主参数解析-提示词
- douyi_category_vector_clean_prompt    # 抖衣-类目向量清洗-提示词
```

### 4.2 Trace 命名规范

```
<业务含义> 而不是 <代码函数名>

好的示例:
- "知衣选品" 而不是 "zhiyi_graph"
- "生成思维链" 而不是 "_think_node"
- "解析主要参数" 而不是 "_main_param_parse_node"
```

### 4.3 异常处理

```python
try:
    messages = coze_loop_client_provider.get_langchain_messages(...)
    result = llm.invoke(messages)
except Exception as e:
    # 1. 记录错误到 trace
    root_span.set_error(e)
    # 2. 记录日志
    logger.exception(f"操作失败 [trace_id={trace_id}]")
    # 3. 使用降级方案（如果有）
    result = fallback_method()
    # 4. 或者抛出业务异常
    raise AppException(...) from e
```

### 4.4 变量设计

提示词变量应该：
- 语义清晰，见名知意
- 类型明确（字符串、数字、列表等）
- 提供默认值或检查必填项
- 文档化（在 CozeLoop 平台上添加说明）

---

## 五、监控指标

### 5.1 提示词相关

- **提示词获取成功率**: 监控 CozeLoop API 可用性
- **提示词版本分布**: 统计各版本使用情况
- **降级触发次数**: 监控降级方案使用频率

### 5.2 Trace 相关

- **工作流执行耗时**: P50, P90, P99
- **工作流成功率**: 成功/失败比例
- **LLM Token 消耗**: 统计成本
- **并发执行数**: 监控系统负载
- **错误类型分布**: 识别主要问题

### 5.3 关键节点监控

- **意图分类耗时**: 影响首屏响应
- **知识库检索耗时**: 可能的性能瓶颈
- **并行任务执行**: 线程池使用情况
- **数据库查询**: 慢查询识别

---

## 六、关键代码文件

| 文件路径 | 作用 |
|---------|------|
| `app/core/clients/coze_loop_client.py` | CozeLoop 客户端封装，提供提示词获取和 trace 上报 |
| `app/core/config/constants.py` | 定义所有提示词 Key 的枚举类 `CozePromptHubKey` |
| `app/service/chains/workflow/base_graph.py` | 工作流基类，统一处理 trace 上报逻辑 |
| `app/config.py` | Apollo 配置管理，动态读取 CozeLoop 相关配置 |
| `app/service/chains/workflow/select/zhiyi_graph.py` | 知衣选品工作流，提示词和 trace 的实际应用示例 |
| `app/service/chains/workflow/main_orchestrator_graph.py` | 主编排工作流，展示降级机制 |

---

## 七、总结

本项目通过 CozeLoop 平台实现了：

1. **提示词工程化**
   - 中心化管理，避免硬编码
   - 支持版本控制和灰度发布
   - 提供降级机制保证系统稳定性

2. **全链路可观测性**
   - 自动捕获 LangChain 所有操作
   - 完整的调用链路追踪
   - 实时监控 LLM 性能和成本

3. **配置中心集成**
   - 与 Apollo 深度集成
   - 支持动态切换提示词版本
   - 环境隔离（gray/prod）

这个架构设计优雅且实用，既提高了开发效率，也便于后期的监控、优化和问题排查。
