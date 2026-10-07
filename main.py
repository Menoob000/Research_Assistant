from agent import app
from langchain_core.messages import HumanMessage

def main():
    message = {'messages' : HumanMessage(content="What's the capital of France")}
    response = app.invoke(message)
    print(response['messages'][-1].content)


if __name__ == "__main__":
    main()
