use ai::{
    chat_completions::{
        ChatCompletion, ChatCompletionMessage, ChatCompletionRequestBuilder,
        ChatCompletionToolFunctionDefinitionBuilder,
    },
    Result,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let openai = ai::clients::openai::Client::from_url("ollama", "http://localhost:11434/v1")?;

    let weather_tool = ChatCompletionToolFunctionDefinitionBuilder::default()
        .name("weather_tool")
        .description("Get current temperature for a given location")
        .parameters(json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Paris, France"
                }
            },
            "required": ["location"],
            "additionalProperties": false
        }))
        .strict(true)
        .build()?
        .into();

    let request = ChatCompletionRequestBuilder::default()
        .model("llama3.2")
        .messages(vec![
            ChatCompletionMessage::System("You are a helpful assistant".into()),
            ChatCompletionMessage::User("What is the weather like in Paris today?".into()),
        ])
        .tools(vec![weather_tool])
        .build()?;

    let response = openai.chat_completions(&request).await?;

    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        for tool_call in tool_calls {
            match tool_call {
                ai::chat_completions::ChatCompletionMessageToolCall::Function { function } => {
                    if function.name == "weather_tool" {
                        let args: serde_json::Value = serde_json::from_str(&function.arguments)?;
                        let location = args["location"].as_str().unwrap();
                        // NOTE: Call weather API to get the current temperature for the location.
                        println!("The current temperature in {location} is 25C.");
                    } else {
                        println!("Unknown tool function: {}", function.name);
                    }
                }
            }
        }
    } else {
        println!("{}", &response.choices[0].message.content.as_ref().unwrap());
    }

    Ok(())
}
