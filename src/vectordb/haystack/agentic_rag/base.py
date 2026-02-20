"""Base classes for agentic RAG pipelines across all vector databases.

Provides shared initialization, routing, and pipeline orchestration.
Subclasses implement database-specific retrieval logic.

Agentic RAG Architecture:

Query Routing:
    Each incoming query is analyzed by an LLM-based router that selects
    the appropriate tool based on query characteristics:
    - Factual lookup → retrieval tool (vector DB search)
    - Current events → web_search tool (external search)
    - Math/logic → calculation tool (LLM reasoning)
    - Complex synthesis → reasoning tool (multi-step analysis)

    The routing decision is made by prompting the LLM with the query
    and available tool descriptions, then parsing the response.

Self-reflection:
    After initial answer generation, the agent evaluates quality:
    - Completeness: Does it address all parts of the query?
    - Accuracy: Is it supported by retrieved documents?
    - Clarity: Is it well-structured and understandable?
    If score < threshold, the agent iterates: retrieves more docs,
    reformulates the answer, and re-evaluates (up to max_iterations).

Multi-hop Retrieval:
    Complex queries requiring multiple facts are decomposed:
    1. Agent identifies sub-questions needed for the answer
    2. Retrieves documents for each sub-question
    3. Synthesizes sub-answers into comprehensive response

Subclasses must implement:
    - _connect(): Database connection logic
    - _create_index(): Collection/index management
    - index_documents(): Document insertion
    - _retrieve(): Vector similarity search
"""

from abc import ABC, abstractmethod
from typing import Any

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.components.agentic_router import AgenticRouter
from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.utils.config import setup_logger


def get_dataloader_instance(config: dict[str, Any]) -> Any:
    """Get dataloader instance from config.

    Args:
        config: Configuration dictionary with dataloader section.

    Returns:
        Dataloader instance.

    Raises:
        ValueError: If dataloader configuration is invalid.
    """
    dataloader_config = config.get("dataloader", {})
    dataset_type = dataloader_config.get("type", "triviaqa")
    dataset_name = dataloader_config.get("dataset_name")
    split = dataloader_config.get("split", "test")
    limit = dataloader_config.get("limit")

    try:
        loader = DataloaderCatalog.create(
            dataset_type,
            split=split,
            limit=limit,
            dataset_id=dataset_name,
        )
        dataset = loader.load()

        class SimpleDataloader:
            """Simple dataloader wrapper."""

            def __init__(self, loaded_dataset) -> None:
                self.loaded_dataset = loaded_dataset

            def load_data(self) -> list[dict[str, Any]]:
                """Return loaded data."""
                return self.loaded_dataset.to_dict_items()

            def get_documents(self) -> list[Document]:
                """Convert data to Haystack Documents."""
                return self.loaded_dataset.to_haystack()

        return SimpleDataloader(dataset)
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_type}: {str(e)}") from e


class BaseAgenticRAGPipeline(ABC):
    """Abstract base class for agentic RAG pipelines.

    Implements the core agentic RAG loop: route → retrieve → generate → reflect.

    Agentic Workflow:
        1. ROUTE: AgenticRouter analyzes query and selects optimal tool
           - Parses query intent using LLM prompting
           - Maps to available tools based on descriptions
           - Default: "retrieval" for direct vector search

        2. RETRIEVE: Execute selected tool to gather context
           - retrieval: Dense similarity search in vector DB
           - web_search: External API calls (placeholder)
           - calculation: LLM-based math/logic solving
           - reasoning: Multi-step synthesis with intermediate steps

        3. GENERATE: LLM creates answer from retrieved context
           - Formats documents into coherent prompt context
           - Uses generator LLM (e.g., Llama 3.3-70B) for response
           - Falls back gracefully if generation fails

        4. REFLECT (optional): Evaluate and refine answer quality
           - AgenticRouter.score_answer() assesses completeness/accuracy
           - If score < quality_threshold: trigger refinement iteration
           - Retrieves additional context or regenerates answer
           - Repeats up to max_iterations times

    Subclass Requirements:
        Subclasses must implement database-specific methods:
        - _connect(): Establish database client connection
        - _create_index(): Initialize collection/index
        - index_documents(): Insert embedded documents
        - _retrieve(): Execute vector similarity search

    Attributes:
        config: Configuration dictionary loaded from YAML.
        logger: Structured logger for pipeline events.
        dataloader: Dataset loader for test/evaluation data.
        dense_embedder: Query embedder for vector search.
        document_embedder: Batch document embedder for indexing.
        router: AgenticRouter for tool selection and reflection.
        generator: LLM client for answer generation.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize agentic RAG pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)
        self._init_embedders()
        self._init_router()
        self._init_generator()
        self._load_dataloader()
        self._connect()
        self._create_index()

    def _init_embedders(self) -> None:
        """Initialize dense embedders from configuration."""
        embeddings_config = self.config.get("embeddings", {})
        dense_model = embeddings_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        batch_size = embeddings_config.get("batch_size", 32)

        model_aliases = {
            "qwen3": "Qwen/Qwen3-Embedding-0.6B",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        dense_model = model_aliases.get(dense_model.lower(), dense_model)

        self.dense_embedder = SentenceTransformersTextEmbedder(model=dense_model)
        self.dense_embedder.warm_up()

        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=dense_model,
            batch_size=batch_size,
        )
        self.document_embedder.warm_up()

        self.logger.info("Initialized dense embedders with model: %s", dense_model)

    def _init_router(self) -> None:
        """Initialize agentic router from configuration."""
        router_config = self.config.get("agentic_rag", {})
        model = router_config.get("model", "llama-3.3-70b-versatile")
        api_key = router_config.get("api_key")

        try:
            self.router = AgenticRouter(model=model, api_key=api_key)
            self.logger.info("Initialized AgenticRouter with model: %s", model)
        except Exception as e:
            self.logger.error("Failed to initialize AgenticRouter: %s", str(e))
            raise

    def _init_generator(self) -> None:
        """Initialize LLM generator from configuration."""
        generator_config = self.config.get("generator", {})
        model = generator_config.get("model", "llama-3.3-70b-versatile")
        api_key = generator_config.get("api_key")
        max_tokens = generator_config.get("max_tokens", 2048)

        import os

        api_key = api_key or os.getenv("GROQ_API_KEY")

        try:
            self.generator = OpenAIGenerator(
                api_key=Secret.from_token(api_key) if api_key else None,
                api_base_url="https://api.groq.com/openai/v1",
                model=model,
                generation_kwargs={"max_tokens": max_tokens},
            )
            self.generator.warm_up()
            self.logger.info("Initialized generator with model: %s", model)
        except Exception as e:
            self.logger.error("Failed to initialize generator: %s", str(e))
            raise

    def _load_dataloader(self) -> None:
        """Initialize dataloader from config.

        Sets up dataloader instance and initializes data containers.
        """
        try:
            self.dataloader = get_dataloader_instance(self.config)
            self.data = None
            self.documents = None
            self.ground_truths = None
            self.logger.info("Initialized dataloader for dataset type")
        except Exception as e:
            self.logger.warning("Failed to initialize dataloader: %s", str(e))
            self.dataloader = None
            self.data = None
            self.documents = None
            self.ground_truths = None

    def load_dataset(
        self, dataset_type: str | None = None, limit: int | None = None
    ) -> None:
        """Load dataset at runtime.

        Args:
            dataset_type: Override config dataloader type (e.g., "triviaqa", "arc").
                         If None, uses config.
            limit: Limit number of samples to load (for quick testing).
        """
        if dataset_type:
            # Update config temporarily
            self.config["dataloader"]["type"] = dataset_type

        if limit:
            # Update limit in config
            self.config["dataloader"]["limit"] = limit

        try:
            self.dataloader = get_dataloader_instance(self.config)
        except Exception as e:
            self.logger.error("Failed to load dataloader: %s", str(e))
            raise

        try:
            self.data = self.dataloader.load_data()

            self.documents = self.dataloader.get_documents()

            # Apply limit if specified
            if limit and self.documents:
                self.documents = self.documents[:limit]

            # Extract ground truths (Q&A pairs)
            self.ground_truths = self._extract_ground_truths()

            self.logger.info(
                "Loaded %d documents with %d ground truths",
                len(self.documents) if self.documents else 0,
                len(self.ground_truths) if self.ground_truths else 0,
            )
        except Exception as e:
            self.logger.error("Failed to load dataset: %s", str(e))
            raise

    def _extract_ground_truths(self) -> list[dict[str, str]]:
        """Extract ground truth Q&A pairs from loaded data.

        Returns:
            List of dicts with 'question' and 'answer' keys.
        """
        if not self.data:
            return []

        ground_truths = []
        for item in self.data:
            if isinstance(item, dict):
                # Handle different dataloader formats
                question = item.get("question") or item.get("query")
                answer = item.get("answer")
                if answer is None:
                    answer = item.get("answers")

                if question and answer is not None:
                    # Handle answer lists
                    if isinstance(answer, list):
                        answer = answer[0] if answer else ""
                    ground_truths.append(
                        {"question": str(question), "answer": str(answer)}
                    )

        return ground_truths

    def _get_routing_enabled(self) -> bool:
        """Check if query routing is enabled.

        Returns:
            True if routing is enabled, False otherwise.
        """
        router_config = self.config.get("agentic_rag", {})
        return router_config.get("routing_enabled", True)

    def _get_self_reflection_enabled(self) -> bool:
        """Check if self-reflection is enabled.

        Returns:
            True if self-reflection is enabled, False otherwise.
        """
        router_config = self.config.get("agentic_rag", {})
        return router_config.get("self_reflection_enabled", False)

    def _get_max_iterations(self) -> int:
        """Get maximum self-reflection iterations.

        Returns:
            Maximum number of refinement iterations.
        """
        router_config = self.config.get("agentic_rag", {})
        return router_config.get("max_iterations", 2)

    def _get_quality_threshold(self) -> int:
        """Get quality threshold for self-reflection.

        Returns:
            Minimum acceptable quality score.
        """
        router_config = self.config.get("agentic_rag", {})
        return router_config.get("quality_threshold", 75)

    @abstractmethod
    def _connect(self) -> None:
        """Establish connection to the vector database.

        Subclasses must implement database-specific connection logic.
        """

    @abstractmethod
    def _create_index(self) -> None:
        """Create or get database index/collection for retrieval.

        Subclasses must implement database-specific index creation.
        """

    def embed_documents(self) -> list[Document]:
        """Embed loaded documents using document embedder.

        Returns:
            List of embedded Document objects with vectors.

        Raises:
            ValueError: If no documents are loaded.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_dataset() first.")

        self.logger.info("Embedding %d documents...", len(self.documents))
        result = self.document_embedder.run(documents=self.documents)
        embedded_docs = result.get("documents", [])
        self.logger.info("Embedded %d documents successfully", len(embedded_docs))
        return embedded_docs

    @abstractmethod
    def index_documents(self) -> int:
        """Index embedded documents into vector database.

        Subclasses implement vectordb-specific insertion logic.

        Returns:
            Number of documents indexed.
        """

    @abstractmethod
    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from vector database.

        Args:
            query: The search query text.
            top_k: Number of results to retrieve.

        Returns:
            List of retrieved Document objects.
        """

    def _generate_answer(
        self,
        query: str,
        documents: list[Document],
    ) -> str:
        """Generate answer from retrieved documents.

        Core answer generation method used by all tool handlers.
        Formats retrieved documents into LLM prompt and generates response.

        Args:
            query: Original user query.
            documents: Retrieved Document objects from vector DB.

        Returns:
            Generated answer string from LLM.
        """
        # Handle empty retrieval case
        if not documents:
            return "No relevant documents found."

        # Build context from top 5 documents (truncation for LLM context limits)
        context = "\n\n".join([doc.content for doc in documents[:5]])

        # Construct standard RAG prompt with context and question
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            # Generate answer using configured LLM
            result = self.generator.run(prompt=prompt)
            replies = result.get("replies", [])
            return replies[0] if replies else "Unable to generate answer."
        except Exception as e:
            self.logger.error("Answer generation failed: %s", str(e))
            return "Answer generation failed."

    def _handle_retrieval(self, query: str, top_k: int) -> dict[str, Any]:
        """Handle retrieval-type query.

        Standard RAG flow: retrieve documents then generate answer.
        This is the default tool when routing is disabled or uncertain.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve.

        Returns:
            Dictionary with documents and generated answer.
        """
        # Retrieve relevant documents from vector database
        documents = self._retrieve(query, top_k)

        # Generate answer using LLM with retrieved context
        answer = self._generate_answer(query, documents)

        return {"documents": documents, "answer": answer, "tool": "retrieval"}

    def _handle_web_search(self, query: str) -> dict[str, Any]:
        """Handle web search-type query.

        Args:
            query: The search query.

        Returns:
            Dictionary with fallback to retrieval or web search result.
        """
        self.logger.info("Web search requested but not implemented, falling back")
        return {
            "documents": [],
            "answer": "Web search not available. Please use direct retrieval.",
            "tool": "web_search",
        }

    def _handle_calculation(self, query: str) -> dict[str, Any]:
        """Handle calculation-type query.

        Routes mathematical/logical queries directly to LLM without
        vector retrieval. The LLM performs step-by-step calculation.

        Args:
            query: The calculation or math query.

        Returns:
            Dictionary with calculation result and empty documents list.
        """
        # Build calculation prompt with explicit step-by-step instruction
        prompt = f"""Solve this problem step by step:

{query}

Provide the calculation steps and final answer."""

        try:
            # Use LLM for calculation (no retrieval needed)
            result = self.generator.run(prompt=prompt)
            replies = result.get("replies", [])
            answer = replies[0] if replies else "Calculation failed."
            return {"documents": [], "answer": answer, "tool": "calculation"}
        except Exception as e:
            self.logger.error("Calculation failed: %s", str(e))
            return {
                "documents": [],
                "answer": "Calculation failed.",
                "tool": "calculation",
            }

    def _handle_reasoning(self, query: str, top_k: int) -> dict[str, Any]:
        """Handle reasoning-type query with multi-hop retrieval.

        Complex queries often require connecting multiple facts. This handler:
        1. Retrieves relevant documents
        2. Guides LLM through structured reasoning steps
        3. Synthesizes comprehensive answer from evidence

        Multi-hop Pattern:
            Queries like "What was the impact of X on Y?" require:
            - Document A: Information about X
            - Document B: Connection between X and Y
            - Document C: Outcome/effect on Y
            The LLM must synthesize across all three.

        Args:
            query: The reasoning query requiring synthesis.
            top_k: Number of documents to retrieve for context.

        Returns:
            Dictionary with documents and reasoned answer.
        """
        # Retrieve initial context for reasoning
        documents = self._retrieve(query, top_k)

        # Build context string from top 5 most relevant documents
        context = "\n\n".join([doc.content for doc in documents[:5]])

        # Construct reasoning prompt with explicit step-by-step instructions
        # This guides the LLM to break down complex queries logically
        prompt = f"""You are a helpful assistant that answers questions using step-by-step reasoning.

Context:
{context}

Question: {query}

Think through this step-by-step:
1. First, identify what information is relevant
2. Then, analyze the key points
3. Finally, synthesize into a comprehensive answer

Answer:"""

        try:
            # Generate structured reasoning response
            result = self.generator.run(prompt=prompt)
            replies = result.get("replies", [])
            answer = replies[0] if replies else "Reasoning failed."
            return {"documents": documents, "answer": answer, "tool": "reasoning"}
        except Exception as e:
            self.logger.error("Reasoning failed: %s", str(e))
            return {
                "documents": documents,
                "answer": "Reasoning failed.",
                "tool": "reasoning",
            }

    def run(
        self,
        query: str,
        top_k: int = 10,
        enable_routing: bool | None = None,
        enable_self_reflection: bool | None = None,
    ) -> dict[str, Any]:
        """Execute the agentic RAG pipeline with routing and reflection.

        This is the main entry point for agentic query processing. It implements
        the full agentic loop: route → execute tool → generate → reflect.

        Agentic Execution Flow:
            1. Check config/runtime flags for routing and reflection
            2. Route: AgenticRouter.select_tool() determines optimal tool
            3. Execute: Call appropriate handler (retrieval/calculation/etc.)
            4. Reflect: If enabled, evaluate and refine answer quality

        Tool Selection Logic:
            The router prompts the LLM with:
            - Query text
            - Available tool descriptions
            - Examples of each query type
            LLM responds with tool name; parsed and validated.

        Self-reflection Logic:
            If reflection enabled and answer exists:
            - Score answer quality (0-100) via AgenticRouter
            - If score >= quality_threshold: return answer
            - If score < threshold and iterations < max:
              * Retrieve additional context
              * Regenerate answer with broader context
              * Re-score and repeat

        Args:
            query: User query text to process.
            top_k: Maximum documents to retrieve per query.
            enable_routing: Override config to force enable/disable routing.
            enable_self_reflection: Override config for reflection.

        Returns:
            Result dictionary containing:
            - documents: List of retrieved Document objects
            - answer: Generated answer string
            - tool: Tool name used (retrieval/web_search/calculation/reasoning)
            - refined: True if answer went through reflection iteration
        """
        # Check runtime overrides or fall back to config settings
        routing = (
            enable_routing
            if enable_routing is not None
            else self._get_routing_enabled()
        )
        reflection = (
            enable_self_reflection
            if enable_self_reflection is not None
            else self._get_self_reflection_enabled()
        )

        self.logger.info(
            "Running agentic RAG: query='%s', routing=%s, reflection=%s",
            query[:50],
            routing,
            reflection,
        )

        try:
            # AGENT STEP 1: Route query to appropriate tool
            tool = self.router.select_tool(query) if routing else "retrieval"

            # AGENT STEP 2: Execute selected tool
            if tool == "retrieval":
                result = self._handle_retrieval(query, top_k)
            elif tool == "web_search":
                result = self._handle_web_search(query)
            elif tool == "calculation":
                result = self._handle_calculation(query)
            elif tool == "reasoning":
                result = self._handle_reasoning(query, top_k)
            else:
                # Unknown tool: fallback to retrieval
                result = self._handle_retrieval(query, top_k)

            # AGENT STEP 3: Self-reflection and iterative refinement
            if reflection and result.get("answer"):
                # Build context string from top 3 retrieved docs
                context = "\n".join(
                    [d.content for d in result.get("documents", [])[:3]]
                )
                # Run reflection loop: score → (refine if needed) → repeat
                refined_answer = self.router.self_reflect_loop(
                    query=query,
                    answer=result["answer"],
                    context=context,
                    max_iterations=self._get_max_iterations(),
                    quality_threshold=self._get_quality_threshold(),
                )
                result["answer"] = refined_answer
                result["refined"] = True

            return result

        except Exception as e:
            self.logger.error("Error during agentic RAG: %s", str(e))
            return {"documents": [], "answer": "Error occurred.", "tool": "error"}

    def evaluate(
        self, questions: list[str] | None = None, ground_truths: list[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate agentic RAG quality using metrics.

        Args:
            questions: Query questions. If None, uses loaded ground_truths.
            ground_truths: Ground truth answers. If None, uses loaded ground_truths.

        Returns:
            Dictionary with evaluation metrics.
        """
        # Use loaded dataset if available
        if questions is None:
            if not self.ground_truths:
                self.logger.warning(
                    "No ground truths loaded. Call load_dataset() first."
                )
                return {"questions": 0, "metrics": {}, "error": "No dataset loaded"}
            questions = [gt["question"] for gt in self.ground_truths]

        self.logger.info(
            "Evaluating agentic RAG pipeline with %d questions", len(questions)
        )

        results = {
            "total": len(questions),
            "queries": [],
            "metrics": {
                "documents_retrieved": 0,
                "tools_used": {},
                "refined_count": 0,
            },
        }

        for i, question in enumerate(questions):
            result = self.run(question)
            results["queries"].append(
                {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "tool": result.get("tool", "unknown"),
                    "documents": len(result.get("documents", [])),
                    "refined": result.get("refined", False),
                }
            )

            # Aggregate metrics
            results["metrics"]["documents_retrieved"] += len(
                result.get("documents", [])
            )
            tool = result.get("tool", "unknown")
            results["metrics"]["tools_used"][tool] = (
                results["metrics"]["tools_used"].get(tool, 0) + 1
            )
            if result.get("refined"):
                results["metrics"]["refined_count"] += 1

            self.logger.debug(
                "Query %d/%d: %s -> %d documents, tool=%s",
                i + 1,
                len(questions),
                question[:50],
                len(result.get("documents", [])),
                result.get("tool", "unknown"),
            )

        # Calculate averages
        if results["total"] > 0:
            results["metrics"]["avg_documents"] = (
                results["metrics"]["documents_retrieved"] / results["total"]
            )
            results["metrics"]["refinement_rate"] = (
                results["metrics"]["refined_count"] / results["total"]
            )

        return results
