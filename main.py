from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

MESSAGE = ("""
You are an assistant retrieving official statistical data from the Internet.
Your task is to find or calculate the number of e-commerce buyers in the United Kingdom by age groups and provide a short methodology note.

### OUTPUT FORMAT (REQUIRED)
Return ONLY a valid Python dict in the format:

{
  "data": {
    "16-24": number,
    "25-34": number,
    "35-44": number,
    "45-54": number,
    "55-64": number,
    "65+": number
  },
  "methodology": "Short note in max 10 sentences explaining how the data was calculated and sources used"
}

No Markdown, no extra comments, no text outside this dict.

### DATA COLLECTION RULES
1. If you cannot find data in the first attempt, retry and consult alternative official sources.
2. If sources provide only percentages (%) of e-commerce buyers, calculate absolute numbers using the latest demographic data for the UK.
3. If available age groups differ, map them logically to the standard groups above.
4. If multiple definitions of e-commerce buyers exist, use the one provided by the source.
5. Never respond with 'data not found'. In case of uncertainty:
   - use alternative official reports,
   - estimate values based on available statistics,
   - interpolate or adjust values using reasonable statistical assumptions.
6. After generating the initial response, perform internal verification. Correct any inconsistencies before returning the final dict.

### SOURCES
Use only official data: Ofcom, ONS, UK Government, OECD, Statista (only if citing official sources).

### GOAL
Return only:
- 'group' dict with numbers of e-commerce buyers by age group
- 'methodology' with a short explanation of calculations and sources
- 'sources' of data used for calculations
""")

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")
# noinspection PyDataclass
class ECommerceBuyersPerAge(BaseModel):
    """Schema for number of e-commerce buyers per age group """
    group: Dict[str, int] = Field(
        default_factory=dict, description="Mapping of age group to the number of e-commerce buyers")
    methodology: str = Field(description="Short methodology note describing how the data was collected, calculated, and sources used")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=ECommerceBuyersPerAge)

def main():
    # print("Hello from langchain-search-agent!")
    result = agent.invoke({"messages": HumanMessage(content=MESSAGE)})
    print(result["structured_response"].answer)


if __name__ == "__main__":
    main()
