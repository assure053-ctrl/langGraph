import asyncio
from mcp.server.fastmcp import FastMCP
from browser_use import Agent
from langchain_ollama import ChatOllama

# MCP 서버 초기화
mcp = FastMCP("browser-use")

# Ollama 설정 (사용 중인 모델명으로 변경하세요. 예: qwen2.5, gemma2)
llm = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11434")

@mcp.tool()
async def browse_the_web(task: str) -> str:
    """
    브라우저를 사용하여 복잡한 웹 과업을 수행합니다.
    예: '네이버에서 최신 AI 뉴스 3개 요약해줘'
    """
    agent = Agent(task=task, llm=llm)
    result = await agent.run()
    return str(result)

if __name__ == "__main__":
    mcp.run()