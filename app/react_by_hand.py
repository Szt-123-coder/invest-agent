"""不用框架、手写的 ReAct 循环，只用来学习。正式运行用的是 agent.py 里 LangChain 的版本。

ReAct = Reasoning + Acting：模型先想该做什么，调用工具，看到结果再想，直到能回答为止。
LangChain 的 create_agent 做的核心事情就是下面这个循环，外加流式输出、出错重试、状态管理等。

运行：python -m app.react_by_hand "澳元最近怎么样"
"""

import sys

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from . import db
from .agent import system_prompt
from .llm import get_model
from .tools import ALL_TOOLS

MAX_STEPS = 6  # 防止模型一直调工具停不下来


def react(question: str) -> str:
    tools = {t.name: t for t in ALL_TOOLS}
    model = get_model().bind_tools(ALL_TOOLS)          # 1. 告诉模型有哪些工具可用
    messages = [SystemMessage(system_prompt()), HumanMessage(question)]
    for step in range(MAX_STEPS):
        reply = model.invoke(messages)                   # 2. 模型思考：直接回答，还是先调工具？
        messages.append(reply)
        if not reply.tool_calls:                         # 3. 不再调工具 = 得出最终回答
            return reply.content
        for call in reply.tool_calls:                    # 4. 真的去执行模型要求的每个工具
            print(f"  第 {step + 1} 步：调用 {call['name']}({call['args']})")
            result = tools[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))  # 5. 把结果交回给模型，回到第 2 步
    return "步骤太多，没能得出结论。"


if __name__ == "__main__":
    db.init_db()
    print(react(sys.argv[1] if len(sys.argv) > 1 else "澳元最近怎么样，要不要换？"))
