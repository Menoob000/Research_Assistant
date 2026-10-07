from langchain_tavily import TavilySearch
from langchain.tools import tool

@tool
def search_tool(query : str):
    """
        A tool to search the web for relevant answers
    """

    search = TavilySearch()
    response = search.invoke(query)
    return response