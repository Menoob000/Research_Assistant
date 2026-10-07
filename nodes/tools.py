from langchain.tools import tool
from langchain_tavily import TavilyResearch, TavilySearch

search_tool = TavilySearch(max_results=3, topic="general")
research_tool = TavilyResearch()