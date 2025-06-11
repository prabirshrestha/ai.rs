use ai::graph::{Graph, END, START};
use ai::Error;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct State {
    message: String,
    count: i32,
    quality_score: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let compiled_graph = Graph::new()
        .add_node("generate_content", |mut state: State| async move {
            state.message = format!("Generated content: {}", state.message);
            state.quality_score = 6; // Initial quality score
            println!("Generate: {}", state.message);
            Ok(state)
        })
        .add_node("improve_content", |mut state: State| async move {
            state.message = format!("Improved: {}", state.message);
            state.quality_score += 3; // Improve quality
            state.count += 1;
            println!(
                "Improve: {} (quality: {})",
                state.message, state.quality_score
            );

            // Example error handling
            if state.count > 5 {
                let custom_error =
                    std::io::Error::new(std::io::ErrorKind::Other, "Too many improvement attempts");
                return Err(Box::new(Error::OtherError(Box::new(custom_error)))
                    as Box<dyn std::error::Error + Send + Sync>);
            }

            Ok(state)
        })
        .add_node("polish_content", |mut state: State| async move {
            state.message = format!("Polished: {}", state.message);
            state.quality_score = 10; // Final quality
            println!(
                "Polish: {} (final quality: {})",
                state.message, state.quality_score
            );
            Ok(state)
        })
        .add_edge(START, "generate_content")
        .add_conditional_edges(
            "generate_content",
            |state: State| async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

                if state.quality_score < 8 {
                    "improve".to_string()
                } else {
                    "polish".to_string()
                }
            },
            {
                let mut mapping = HashMap::new();
                mapping.insert("improve", "improve_content");
                mapping.insert("polish", "polish_content");
                mapping
            },
        )
        .add_edge("improve_content", "polish_content")
        .add_edge("polish_content", END)
        .compile()?;

    let initial_state = State {
        message: "Hello World".to_string(),
        count: 0,
        quality_score: 0,
    };

    let result = compiled_graph.execute(initial_state).await?;
    println!("Sequential Final result: {:?}", result);

    Ok(())
}
