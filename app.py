import streamlit as st
import json
import os
import pandas as pd
from rag.ingest import ingest_pdf
from rag.embeddings import get_embedder
from rag.vector_store import build_vector_store
from agent.agent import build_agent
from agent.tools import initialize_tools
from llm.groq_llm import get_llm

def extract_final_answer(agent_response):
    messages = agent_response.get("messages", [])

    # Traverse messages in reverse to find the final AI answer
    for msg in reversed(messages):
        if msg.type == "ai" and not msg.tool_calls:
            return msg.content

    return messages


st.set_page_config(page_title="BrainAI", layout="wide")
st.title("🧠 BrainAI — Autonomous Research & Intelligence Agent")

# Navigation tabs
tab1, tab2 = st.tabs(["🔍 Research Workspace", "📊 Evaluation Dashboard"])

with tab1:
    st.subheader("Document Q&A Pipeline")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        chunks = ingest_pdf(uploaded_file)
        st.success(f"Document ingested: {len(chunks)} chunks")

        embedder = get_embedder()
        vector_store = build_vector_store(chunks, embedder)

        llm = get_llm()
        initialize_tools(vector_store, llm)

        agent = build_agent()

        user_goal = st.text_input(
            "Enter your goal",
            placeholder="Summarize the document and give risks with mitigations"
        )

        if st.button("Run Agent"):
            with st.spinner("Agent planning and executing..."):
                response = agent.invoke({
                    "messages": [
                        {"role": "user", "content": user_goal}
                    ]
                })

            final_answer = extract_final_answer(response)
            st.subheader("📌 BrainAI Output")
            st.markdown(final_answer)

with tab2:
    st.subheader("📈 RAGAS Evaluation Harness")
    results_path = "ragas_results.json"
    
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            
            summary = eval_results.get("summary", {})
            details = eval_results.get("details", [])
            
            # Display overall metrics
            st.markdown("### Overall System Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            faith = summary.get("faithfulness")
            col1.metric(
                label="Faithfulness",
                value=f"{faith:.2%}" if faith is not None else "N/A",
                help="Groundedness: Is the generated answer based strictly on the retrieved context?"
            )
            
            rel = summary.get("answer_relevancy")
            col2.metric(
                label="Answer Relevancy",
                value=f"{rel:.2%}" if rel is not None else "N/A",
                help="Relevancy: Does the generated answer address the user's question directly?"
            )
            
            prec = summary.get("context_precision")
            col3.metric(
                label="Context Precision",
                value=f"{prec:.2%}" if prec is not None else "N/A",
                help="Precision: Are the retrieved context chunks highly relevant to the query?"
            )
            
            rec = summary.get("context_recall")
            col4.metric(
                label="Context Recall",
                value=f"{rec:.2%}" if rec is not None else "N/A",
                help="Recall: Did the vector search retrieve all the key facts from the ground truth?"
            )
            
            # Plot metrics comparison
            metrics_df = pd.DataFrame(
                [{"Metric": k.replace("_", " ").title(), "Score (%)": v * 100} for k, v in summary.items()]
            )
            st.markdown("---")
            st.markdown("### Metrics Comparison")
            st.bar_chart(metrics_df.set_index("Metric"))
            
            # Detailed breakdown table
            st.markdown("---")
            st.markdown("### Detailed Evaluation Runs")
            
            table_data = []
            for item in details:
                table_data.append({
                    "Question": item["question"],
                    "Faithfulness": f"{item['faithfulness']:.2f}" if item['faithfulness'] is not None else "N/A",
                    "Answer Relevancy": f"{item['answer_relevancy']:.2f}" if item['answer_relevancy'] is not None else "N/A",
                    "Context Precision": f"{item['context_precision']:.2f}" if item['context_precision'] is not None else "N/A",
                    "Context Recall": f"{item['context_recall']:.2f}" if item['context_recall'] is not None else "N/A",
                })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            
            # Expanders for individual inspection
            st.markdown("### Question-by-Question Deep Dive")
            for i, item in enumerate(details):
                with st.expander(f"Q{i+1}: {item['question']}"):
                    st.markdown(f"**🎯 Ground Truth:** {item['ground_truth']}")
                    st.markdown(f"**🤖 Generated Answer:** {item['answer']}")
                    
                    st.markdown("**📂 Retrieved Contexts:**")
                    for c_idx, ctx in enumerate(item['contexts']):
                        st.caption(f"Chunk {c_idx+1}:")
                        st.info(ctx)
                        
                    # Individual scores
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Faithfulness", f"{item['faithfulness']:.2f}" if item['faithfulness'] is not None else "N/A")
                    sc2.metric("Answer Relevancy", f"{item['answer_relevancy']:.2f}" if item['answer_relevancy'] is not None else "N/A")
                    sc3.metric("Context Precision", f"{item['context_precision']:.2f}" if item['context_precision'] is not None else "N/A")
                    sc4.metric("Context Recall", f"{item['context_recall']:.2f}" if item['context_recall'] is not None else "N/A")
                    
        except Exception as e:
            st.error(f"Error loading evaluation results: {e}")
            
    else:
        st.info("💡 **No evaluation results found.**")
        st.markdown(
            """
            To run the evaluation harness and generate the performance metrics dashboard:
            
            1. Make sure you have configured your environment with `GROQ_API_KEY`.
            2. Run the evaluation script in your terminal:
               ```bash
               python evaluate_rag.py
               ```
            3. Once the execution completes, refresh this page to see the computed metrics and detailed dashboard!
            """
        )
