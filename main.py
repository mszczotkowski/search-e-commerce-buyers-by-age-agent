from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from langsmith import uuid7, Client
# client=Client()
# run_id = uuid7()
# client.create_run(
#     id=run_id,
#     name="test-run"
# )
# MESSAGE = "What is the weather in Tokyo?"
MESSAGE = ("Jesteś asystentem pomagającym w znalezieniu danych statystycznych w Internecie."
           "Twoim zadaniem jest wyszukanie danych o liczbie e-commerce buyers w United Kingdom."
           "e-commerce buyers to osoby, które w ostatnich 12 miesiącach kupiły lub zamówiły towary lub usługi przez internet."
           "Dane powinny pochodzić z oficjalnych źródeł."
           "Liczba e-commerce buyers powinna być podzielona na grupy wiekowe."
           "Jeśli znajdziesz jedynie odsetek (%) e-commerce buyers w różnych grupach wiekowych znajdź również łączną liczbę "
           "osób w danym przedziale wiekowym i przemnóż odsetek e-buyers przez liczbę osób w danej grupie wiekowej"
           "Zastosuj najświerzsze dane dotyczące struktury ludności oraz najświerzsze dane dotyczące odsetka e-commerce buyers."
           "Jeśli nie znajdziesz wiarygodnych danych dla United Kingdom znajdź dane jedynie dla Great Britan."
           "To już wszystkie instrukcje jakie możesz otrzymać. Znajdź najdokładniejsze informacje.")


# noinspection PyDataclass
class ECommerceBuyersPerAge(BaseModel):
    """Schema for number of e-commerce buyers per age group """
    group: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping: age group to the number of e-commerce buyers")

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources and """
    answer: str = Field(description="The agent's answer to the query")
    # noinspection PyDataclass
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")
    # ecb_list: ECommerceBuyersPerAge = Field(description="Dict with age groups and number of e-commerce buyers")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    # print("Hello from langchain-search-agent!")
    result = agent.invoke({"messages": HumanMessage(content=MESSAGE)})
    print(result["structured_response"].answer)


if __name__ == "__main__":
    main()
