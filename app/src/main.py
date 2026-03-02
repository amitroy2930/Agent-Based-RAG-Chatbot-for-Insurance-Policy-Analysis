import os
import re
import json
import yaml
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # beginner-friendly local memory checkpointer :contentReference[oaicite:3]{index=3}


# ----------------------------
# 1) Utilities
# ----------------------------

def clean_json_response(text: str) -> str:
    """Extract JSON from fenced ```json ...``` if present; otherwise return stripped text."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# ----------------------------
# 2) Config + Prompt Loading (OOP)
# ----------------------------

@dataclass
class RAGConfig:
    chat_model: str = "gpt-5-mini"
    temperature: float = 1.0
    max_tokens: int = 8192

    embedding_model: str = "text-embedding-3-small"

    chroma_dir: str = "../data/vector_data/chroma_db"
    chroma_collection: str = "policy_docs"

    prompt_path: str = "./utils/prompt_templates.yaml"
    output_dir: str = "../data/generated_output_test"

class PromptRegistry:
    def __init__(self, prompt_path: str):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def get(self, key: str) -> str:
        if key not in self.prompts:
            raise KeyError(f"Prompt '{key}' not found in YAML.")
        return self.prompts[key]


# ----------------------------
# 3) Core Services (OOP)
# ----------------------------

class LLMService:
    def __init__(self, cfg: RAGConfig):
        self.llm = ChatOpenAI(
            model=cfg.chat_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
        )

    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content.strip()

class VectorStoreService:
    def __init__(self, cfg: RAGConfig):
        embedding_model = OpenAIEmbeddings(
            model=cfg.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
        )
        self.store = Chroma(
            persist_directory=cfg.chroma_dir,
            collection_name=cfg.chroma_collection,
            embedding_function=embedding_model,
        )

    def retrieve(self, query: str, policy: Optional[str] = None, plan: Optional[str] = None, k: int = 5):
        filter_conditions = []
        if policy:
            filter_conditions.append({"company": {"$eq": policy}})
        if plan:
            filter_conditions.append({"plan": {"$eq": plan}})

        if len(filter_conditions) > 1:
            combined_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            combined_filter = filter_conditions[0]
        else:
            combined_filter = None

        retriever = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "filter": combined_filter},
        )
        return retriever.invoke(query)


class QueryRestructurer:
    def __init__(self, llm: LLMService, prompts: PromptRegistry):
        self.llm = llm
        self.prompts = prompts

    def run(self, original_query: str) -> Dict[str, Any]:
        prompt = self.prompts.get("adaptive_rag_prompt").format(original_query=original_query)
        raw = self.llm.invoke(prompt)
        try:
            return json.loads(clean_json_response(raw))
        except json.JSONDecodeError:
            # Keep it simple for beginners: fall back to minimal structure
            return {"query": original_query, "policy": None, "plan": None}


class DocumentRefiner:
    """Filters docs using your 'corrective_rag_prompt' grader."""
    def __init__(self, llm: LLMService, prompts: PromptRegistry):
        self.llm = llm
        self.prompts = prompts

    @staticmethod
    def _doc_to_entry(doc) -> str:
        return (
            f'"company": {doc.metadata.get("company")}\n'
            f'"plan": {doc.metadata.get("plan")}\n'
            f'"content": {doc.page_content}'
        )

    def grade_doc_entry(self, original_query: str, doc_entry: str) -> int:
        prompt = self.prompts.get("corrective_rag_prompt").format(
            original_query=original_query,
            retrieved_doc=doc_entry,
        )
        score_text = self.llm.invoke(prompt)
        # Be defensive
        try:
            return int(re.search(r"-?\d+", score_text).group())
        except Exception:
            return 0

    def run(self, original_query: str, docs) -> str:
        doc_entries = [self._doc_to_entry(d) for d in docs]
        kept: List[str] = []
        for entry in doc_entries:
            if self.grade_doc_entry(original_query, entry) >= 2:
                kept.append(entry)
        return "\n\n".join(kept)


class AnswerGenerator:
    def __init__(self, llm: LLMService, prompts: PromptRegistry):
        self.llm = llm
        self.prompts = prompts

    def run(self, original_query: str, refined_docs: str, history: List[BaseMessage]) -> str:
        prompt = self.prompts.get("final_answer_generation_prompt").format(
            original_query=original_query,
            retrieved_docs=refined_docs,
            history=history
        )
        res =  self.llm.invoke(prompt)
        return res.strip()


class HallucinationGrader:
    """Returns True if grounded, False otherwise."""
    def __init__(self, llm: LLMService, prompts: PromptRegistry):
        self.llm = llm
        self.prompts = prompts

    def run(self, documents: str, answer: str, history) -> bool:
        prompt = self.prompts.get("hallucination_checker_prompt").format(
            docs=documents,
            answer=answer,
            history=history
        )
        raw = self.llm.invoke(prompt).strip()
        # IMPORTANT FIX: compare value, do not rely on truthiness of a string.
        return raw == "1"


class OutputWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_markdown(self, query: str, content: str) -> str:
        path = os.path.join(self.output_dir, f"{safe_filename(query)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


# ----------------------------
# 4) LangGraph State + Graph (OOP)
# ----------------------------

class RAGState(TypedDict):
    history: Annotated[List[BaseMessage], add_messages]  # reducer-based conversation history
    query: str
    refined_query: Dict[str, Any]
    docs: Any
    refined_docs: str
    answer: str
    grounded: bool
    output_path: str


class RAGApp:
    """
    Beginner-friendly:
    - build_graph() wires nodes
    - invoke(query) runs the graph
    """

    def __init__(self, cfg: RAGConfig):
        load_dotenv()

        self.cfg = cfg
        self.prompts = PromptRegistry(cfg.prompt_path)
        self.llm = LLMService(cfg)
        self.vs = VectorStoreService(cfg)

        self.restructurer = QueryRestructurer(self.llm, self.prompts)
        self.refiner = DocumentRefiner(self.llm, self.prompts)
        self.generator = AnswerGenerator(self.llm, self.prompts)
        self.hallucination = HallucinationGrader(self.llm, self.prompts)
        self.writer = OutputWriter(cfg.output_dir)

        self.graph = self.build_graph()

    # ---- Graph nodes (each takes state, returns partial state update) ----

    def node_restructure(self, state: RAGState) -> Dict[str, Any]:
        rq = self.restructurer.run(state["query"])
        return {"refined_query": rq}

    def node_retrieve(self, state: RAGState) -> Dict[str, Any]:
        rq = state["refined_query"]

        # Your original logic supports either:
        # - {"query": "...", "policy": "...", "plan": "..."}
        # - or multiple subqueries in dict values
        docs = []

        if rq.get("query"):
            docs.extend(self.vs.retrieve(rq["query"], rq.get("policy"), rq.get("plan"), k=5))
        else:
            for v in rq.values():
                if isinstance(v, dict) and v.get("query"):
                    docs.extend(self.vs.retrieve(v["query"], v.get("policy"), v.get("plan"), k=5))

        return {"docs": docs}

    def node_refine_docs(self, state: RAGState) -> Dict[str, Any]:
        refined_docs = self.refiner.run(state["query"], state["docs"])
        return {"refined_docs": refined_docs}

    def node_generate(self, state: RAGState) -> Dict[str, Any]:
        answer = self.generator.run(state["query"], state.get("refined_docs", ""), state.get("history", []))
        return {"answer": answer}

    def node_grade(self, state: RAGState) -> Dict[str, Any]:
        grounded = self.hallucination.run(state.get("refined_docs", ""), state["answer"], state.get("history"))
        return {"grounded": grounded}

    def node_persist(self, state: RAGState) -> Dict[str, Any]:
        path = self.writer.write_markdown(state["query"], state["answer"])
        return {
            "history": [
                HumanMessage(content=state["query"]),
                AIMessage(content=state["answer"]),
            ],
            "output_path": path,
        }

    # ---- Graph wiring ----

    def build_graph(self):
        g = StateGraph(RAGState)

        g.add_node("restructure", self.node_restructure)
        g.add_node("retrieve", self.node_retrieve)
        g.add_node("refine_docs", self.node_refine_docs)
        g.add_node("generate", self.node_generate)
        g.add_node("grade", self.node_grade)
        g.add_node("persist", self.node_persist)

        g.add_edge(START, "restructure")
        g.add_edge("restructure", "retrieve")
        g.add_edge("retrieve", "refine_docs")
        g.add_edge("refine_docs", "generate")
        g.add_edge("generate", "grade")

        # Conditional routing based on hallucination grade
        def route_after_grade(state: RAGState) -> str:
            return "persist" if state.get("grounded") else END

        g.add_conditional_edges("grade", route_after_grade, {"persist": "persist", END: END})
        g.add_edge("persist", END)

        # Beginner-friendly checkpointer: keeps state if you want to inspect / debug locally :contentReference[oaicite:4]{index=4}
        return g.compile(checkpointer=MemorySaver())

    def invoke(self, query: str, thread_id: str):
        return self.graph.invoke(
            {"query": query},
            config={"configurable": {"thread_id": thread_id}}
        )


# ----------------------------
# 5) Run it
# ----------------------------

if __name__ == "__main__":
    cfg = RAGConfig()
    app = RAGApp(cfg)

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = app.invoke(query, "default_thread")

        print("\n--- RESULT ---")
        print("Grounded:", result.get("grounded"))
        if result.get("grounded"):
            print("Saved to:", result.get("output_path"))
        else:
            print("Not saved (answer not grounded).")
        print("\nAnswer:\n", result.get("answer", ""))