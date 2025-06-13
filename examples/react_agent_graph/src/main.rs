// Based on https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae
// Graph-based implementation inspired by https://github.com/langchain-ai/react-agent

use ai::{
    chat_completions::{
        ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall,
        ChatCompletionRequestBuilder, ChatCompletionTool,
        ChatCompletionToolFunctionDefinitionBuilder, FinishReason,
    },
    graph::{Graph, END, START},
    Result,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub messages: Vec<ChatCompletionMessage>,
    pub pending_tool_calls: Vec<ChatCompletionMessageToolCall>,
    pub iteration: usize,
    pub max_iterations: usize,
    pub final_answer: Option<String>,
}

impl AgentState {
    pub fn new(query: &str, max_iterations: usize) -> Self {
        let messages = vec![
            ChatCompletionMessage::System(
                r#"/no_think You are a ReAct (Reasoning and Acting) agent. When you need information to answer a question, use the available tools to gather that information.

Think step by step:
1. Analyze what information you need
2. Use tools to gather the required information
3. Continue until you have enough information to provide a complete answer
4. Provide a comprehensive final answer

Be thorough and make sure to gather all necessary facts before concluding."#.into()
            ),
            ChatCompletionMessage::User(format!("Question: {}", query).into()),
        ];

        Self {
            messages,
            pending_tool_calls: Vec::new(),
            iteration: 0,
            max_iterations,
            final_answer: None,
        }
    }
}

pub struct ReactAgentGraph {
    client: Arc<dyn ChatCompletion>,
    tools: Vec<ChatCompletionTool>,
}

impl ReactAgentGraph {
    pub fn new(client: Box<dyn ChatCompletion>) -> Self {
        let tools = vec![
            ChatCompletionToolFunctionDefinitionBuilder::default()
                .name("web_search".to_string())
                .description("Search the web for information about people, events, or facts. Useful for finding current information, biographies, birth dates, and other factual data.".to_string())
                .parameters(json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find information about people, events, or facts."
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }))
                .strict(true)
                .build()
                .unwrap()
                .into()
        ];

        Self {
            client: Arc::from(client),
            tools,
        }
    }

    async fn execute_web_search(&self, query: &str) -> Result<String> {
        // Dummy simulation based on query content
        let query_lower = query.to_lowercase();

        if query_lower.contains("ronaldo") {
            Ok(
                "Cristiano Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal."
                    .to_string(),
            )
        } else if query_lower.contains("messi") {
            Ok("Lionel Messi was born on June 24, 1987, in Rosario, Argentina.".to_string())
        } else if query_lower.contains("birth")
            || query_lower.contains("age")
            || query_lower.contains("older")
        {
            Ok("Search results: Cristiano Ronaldo - Born February 5, 1985. Lionel Messi - Born June 24, 1987".to_string())
        } else {
            Ok(format!(
                "Search simulation: No specific information found for query '{}'",
                query
            ))
        }
    }

    async fn call_model(
        &self,
        mut state: AgentState,
    ) -> std::result::Result<AgentState, Box<dyn std::error::Error + Send + Sync>> {
        state.iteration += 1;
        println!("--- Iteration {} ---", state.iteration);

        if state.iteration > state.max_iterations {
            return Err(Box::new(ai::Error::UnknownError(
                "Maximum iterations reached without finding answer".to_string(),
            )));
        }

        let request = ChatCompletionRequestBuilder::default()
            .model("qwen3")
            .messages(state.messages.clone())
            .tools(self.tools.clone())
            .temperature(0.0)
            .build()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        // Use streaming to show real-time thinking
        let mut stream = self
            .client
            .stream_chat_completions(&request)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
        let mut full_content = String::new();
        let mut tool_calls = Vec::new();
        let mut finish_reason = None;

        print!("💭 Agent thinking: ");
        io::stdout()
            .flush()
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            if !chunk.choices.is_empty() {
                let choice = &chunk.choices[0];

                // Stream content
                if let Some(content) = &choice.delta.content {
                    print!("{}", content);
                    io::stdout()
                        .flush()
                        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
                    full_content.push_str(content);
                }

                // Collect tool calls
                if let Some(delta_tool_calls) = &choice.delta.tool_calls {
                    for tool_call in delta_tool_calls {
                        tool_calls.push(tool_call.clone());
                    }
                }

                // Check finish reason
                if let Some(reason) = choice.finish_reason {
                    finish_reason = Some(reason);
                }
            }
        }

        println!(); // New line after streaming

        // Add the assistant message to the conversation
        if !tool_calls.is_empty() {
            println!("🔧 Tool calls requested: {}", tool_calls.len());

            let assistant_content = if !full_content.is_empty() {
                Some(
                    ai::chat_completions::ChatCompletionAssistantMessageContent::Text(
                        full_content.clone(),
                    ),
                )
            } else {
                None
            };

            state.messages.push(ChatCompletionMessage::Assistant(
                ai::chat_completions::ChatCompletionAssistantMessage {
                    content: assistant_content,
                    refusal: None,
                    name: None,
                },
            ));

            state.pending_tool_calls = tool_calls;
        } else if !full_content.is_empty() {
            // No tool calls, this might be the final answer
            if matches!(finish_reason, Some(FinishReason::Stop)) {
                println!("✅ Final answer received");
                state.final_answer = Some(full_content);
            } else {
                // Continue the conversation
                state
                    .messages
                    .push(ChatCompletionMessage::Assistant(full_content.into()));
                state.messages.push(ChatCompletionMessage::User(
                    "Please continue your analysis or use tools if you need more information."
                        .into(),
                ));
            }
        } else {
            state.messages.push(ChatCompletionMessage::User(
                "Please provide your analysis or use tools to gather information.".into(),
            ));
        }

        Ok(state)
    }

    async fn execute_tools(
        &self,
        mut state: AgentState,
    ) -> std::result::Result<AgentState, Box<dyn std::error::Error + Send + Sync>> {
        // Execute each tool call
        for tool_call in &state.pending_tool_calls {
            match tool_call {
                ChatCompletionMessageToolCall::Function { function } => {
                    println!("📞 Calling tool: {}", function.name);
                    println!("📋 Arguments: {}", function.arguments);

                    let result = match function.name.as_str() {
                        "web_search" => {
                            let args: serde_json::Value = serde_json::from_str(&function.arguments)
                                .map_err(|e| {
                                    Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                                })?;
                            let query = args["query"].as_str().ok_or_else(|| {
                                Box::new(ai::Error::UnknownError(
                                    "Missing query parameter".to_string(),
                                ))
                                    as Box<dyn std::error::Error + Send + Sync>
                            })?;
                            self.execute_web_search(query).await.map_err(|e| {
                                Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                            })?
                        }
                        _ => format!("Unknown tool: {}", function.name),
                    };

                    println!("📊 Tool result: {}", result);

                    // Add tool result as a user message
                    state.messages.push(ChatCompletionMessage::User(
                        format!("Tool '{}' returned: {}", function.name, result).into(),
                    ));
                }
            }
        }

        // Clear pending tool calls after execution
        state.pending_tool_calls.clear();
        println!();

        Ok(state)
    }

    pub async fn run(&self, query: &str) -> Result<String> {
        let initial_state = AgentState::new(query, 10);

        // Build the graph
        let graph = Graph::new();

        let graph = graph
            .add_node("call_model", {
                let agent = self.clone();
                move |state: AgentState| {
                    let agent = agent.clone();
                    Box::pin(async move { agent.call_model(state).await })
                }
            })
            .add_node("execute_tools", {
                let agent = self.clone();
                move |state: AgentState| {
                    let agent = agent.clone();
                    Box::pin(async move { agent.execute_tools(state).await })
                }
            })
            .add_edge(START, "call_model")
            .add_conditional_edges(
                "call_model",
                |state: AgentState| {
                    Box::pin(async move {
                        if state.final_answer.is_some() {
                            END.to_string()
                        } else if !state.pending_tool_calls.is_empty() {
                            "execute_tools".to_string()
                        } else {
                            "call_model".to_string()
                        }
                    })
                },
                {
                    let mut mapping = HashMap::new();
                    mapping.insert("execute_tools", "execute_tools");
                    mapping.insert("call_model", "call_model");
                    mapping.insert(END, END);
                    mapping
                },
            )
            .add_edge("execute_tools", "call_model");

        let compiled_graph = graph
            .compile()
            .map_err(|e| ai::Error::UnknownError(format!("Graph compilation failed: {}", e)))?;

        // Display the Mermaid graph
        println!("📊 Graph Structure (Mermaid):");
        println!("{}", compiled_graph.draw_mermaid());
        println!("🔗 View at: https://mermaid.live\n");

        // Execute the graph
        let final_state = compiled_graph
            .execute(initial_state)
            .await
            .map_err(|e| ai::Error::UnknownError(format!("Graph execution failed: {}", e)))?;

        // Return the final answer
        final_state
            .final_answer
            .ok_or_else(|| ai::Error::UnknownError("No final answer was generated".to_string()))
    }
}

impl Clone for ReactAgentGraph {
    fn clone(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            tools: self.tools.clone(),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🤖 ReAct Agent Graph: Who is older, Cristiano Ronaldo or Lionel Messi?\n");

    let client = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
    let agent = ReactAgentGraph::new(Box::new(client));

    let query = "Who is older, Cristiano Ronaldo or Lionel Messi?";

    match agent.run(query).await {
        Ok(answer) => {
            println!("🎯 Final Answer:");
            println!("{}", answer);
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }

    Ok(())
}
