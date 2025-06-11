use ai::graph::{Graph, NodeResult, END};
use ai::Error;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct State {
    message: String,
    count: i32,
}

async fn greet_node(mut state: State) -> NodeResult<State> {
    state.message = format!("Hello, {}!", state.message);
    println!("Greet: {}", state.message);
    Ok(state)
}

async fn count_node(mut state: State) -> NodeResult<State> {
    state.count += 1;
    println!("Count incremented to: {}", state.count);

    // Example of using OtherError for demonstration
    if state.count == 5 {
        let custom_error = std::io::Error::new(std::io::ErrorKind::Other, "Count reached 5");
        return Err(Box::new(Error::OtherError(Box::new(custom_error))));
    }

    Ok(state)
}

async fn decision_node(state: State) -> NodeResult<State> {
    println!("Decision: count is {}", state.count);
    Ok(state)
}

async fn final_node(mut state: State) -> NodeResult<State> {
    state.message = format!("{} (Final)", state.message);
    println!("Final: {}", state.message);
    Ok(state)
}

fn should_continue(state: &State) -> String {
    if state.count < 3 {
        "continue".to_string()
    } else {
        "end".to_string()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut graph = Graph::new();

    graph
        .add_node("greet", greet_node)
        .add_node("count", count_node)
        .add_node("decision", decision_node)
        .add_node("final", final_node);

    graph.add_edge("greet", "count");

    let mut conditional_mapping = HashMap::new();
    conditional_mapping.insert("continue".to_string(), "count".to_string());
    conditional_mapping.insert("end".to_string(), END.to_string());

    graph
        .add_edge("count", "decision")
        .add_conditional_edges("decision", should_continue, conditional_mapping)
        .set_entry_point("greet")
        .set_finish_point("final");

    let compiled_graph = graph.compile()?;

    let initial_state = State {
        message: "World".to_string(),
        count: 0,
    };

    let result = compiled_graph.execute(initial_state).await?;
    println!("Sequential Final result: {:?}", result);

    Ok(())
}
