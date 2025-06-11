// Based on https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae

use ai::{
    chat_completions::{
        ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall,
        ChatCompletionRequestBuilder, ChatCompletionTool,
        ChatCompletionToolFunctionDefinitionBuilder, FinishReason,
    },
    Result,
};
use futures::StreamExt;
use serde_json::json;
use std::io::{self, Write};

pub struct ReactAgent {
    client: Box<dyn ChatCompletion>,
    tools: Vec<ChatCompletionTool>,
    max_iterations: usize,
}

impl ReactAgent {
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
            client,
            tools,
            max_iterations: 10,
        }
    }

    async fn execute_web_search(&self, query: &str) -> Result<String> {
        // Dummy simulation based on query content
        let query_lower = query.to_lowercase();

        if query_lower.contains("ronaldo") {
            Ok("Cristiano Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal. He is currently 39 years old.".to_string())
        } else if query_lower.contains("messi") {
            Ok("Lionel Messi was born on June 24, 1987, in Rosario, Argentina. He is currently 37 years old.".to_string())
        } else if query_lower.contains("birth")
            || query_lower.contains("age")
            || query_lower.contains("older")
        {
            Ok("Search results: Cristiano Ronaldo - Born February 5, 1985 (age 39). Lionel Messi - Born June 24, 1987 (age 37).".to_string())
        } else {
            Ok(format!(
                "Search simulation: No specific information found for query '{}'",
                query
            ))
        }
    }

    pub async fn run(&self, query: &str) -> Result<String> {
        let mut messages = vec![
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

        for iteration in 0..self.max_iterations {
            println!("--- Iteration {} ---", iteration + 1);

            let request = ChatCompletionRequestBuilder::default()
                .model("qwen3")
                .messages(messages.clone())
                .tools(self.tools.clone())
                .temperature(0.0)
                .build()?;

            // Use streaming to show real-time thinking
            let mut stream = self.client.stream_chat_completions(&request).await?;
            let mut full_content = String::new();
            let mut tool_calls = Vec::new();
            let mut finish_reason = None;

            print!("💭 Agent thinking: ");
            io::stdout().flush()?;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                if !chunk.choices.is_empty() {
                    let choice = &chunk.choices[0];

                    // Stream content
                    if let Some(content) = &choice.delta.content {
                        print!("{}", content);
                        io::stdout().flush()?;
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

            // Check if the model wants to use tools
            if !tool_calls.is_empty() {
                println!("🔧 Tool calls requested: {}", tool_calls.len());

                // Add the assistant message with tool calls to the conversation
                let assistant_content = if !full_content.is_empty() {
                    Some(
                        ai::chat_completions::ChatCompletionAssistantMessageContent::Text(
                            full_content.clone(),
                        ),
                    )
                } else {
                    None
                };

                messages.push(ChatCompletionMessage::Assistant(
                    ai::chat_completions::ChatCompletionAssistantMessage {
                        content: assistant_content,
                        refusal: None,
                        name: None,
                    },
                ));

                // Execute each tool call
                for tool_call in &tool_calls {
                    match tool_call {
                        ChatCompletionMessageToolCall::Function { function } => {
                            println!("📞 Calling tool: {}", function.name);
                            println!("📋 Arguments: {}", function.arguments);

                            let result = match function.name.as_str() {
                                "web_search" => {
                                    let args: serde_json::Value =
                                        serde_json::from_str(&function.arguments)?;
                                    let query = args["query"].as_str().ok_or_else(|| {
                                        ai::Error::UnknownError(
                                            "Missing query parameter".to_string(),
                                        )
                                    })?;
                                    self.execute_web_search(query).await?
                                }
                                _ => format!("Unknown tool: {}", function.name),
                            };

                            println!("📊 Tool result: {}", result);

                            // Add tool result as a user message
                            messages.push(ChatCompletionMessage::User(
                                format!("Tool '{}' returned: {}", function.name, result).into(),
                            ));
                        }
                    }
                }
            } else {
                // No tool calls, check if we have a final answer
                if !full_content.is_empty() {
                    // Check finish reason to determine if this is the final answer
                    if matches!(finish_reason, Some(FinishReason::Stop)) {
                        println!("✅ Final answer received");
                        return Ok(full_content);
                    } else {
                        // Continue the conversation
                        messages.push(ChatCompletionMessage::Assistant(full_content.into()));
                        messages.push(ChatCompletionMessage::User(
                            "Please continue your analysis or use tools if you need more information.".into(),
                        ));
                    }
                } else {
                    messages.push(ChatCompletionMessage::User(
                        "Please provide your analysis or use tools to gather information.".into(),
                    ));
                }
            }

            println!();
        }

        Err(ai::Error::UnknownError(
            "Maximum iterations reached without finding answer".to_string(),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🤖 ReAct Agent: Who is older, Cristiano Ronaldo or Lionel Messi?\n");

    let client = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;
    let agent = ReactAgent::new(Box::new(client));

    let query = "Who is older, Cristiano Ronaldo or Lionel Messi? I need their birth dates to determine this.";

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
