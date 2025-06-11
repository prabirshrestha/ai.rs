use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

pub const START: &str = "__start__";
pub const END: &str = "__end__";

pub type NodeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type NodeFn<T> = Box<dyn Fn(T) -> Pin<Box<dyn Future<Output = NodeResult<T>> + Send>> + Send + Sync>;
pub type ConditionalFn<T> = Box<dyn Fn(&T) -> String + Send + Sync>;

#[derive(Clone)]
pub struct Edge {
    pub from: String,
    pub to: String,
}

pub struct ConditionalEdge<T> {
    pub from: String,
    pub condition: ConditionalFn<T>,
    pub mapping: HashMap<String, String>,
}

pub struct Graph<T> {
    nodes: HashMap<String, NodeFn<T>>,
    edges: Vec<Edge>,
    conditional_edges: Vec<ConditionalEdge<T>>,
    entry_point: Option<String>,
    finish_point: Option<String>,
}

impl<T> Default for Graph<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Graph<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            conditional_edges: Vec::new(),
            entry_point: None,
            finish_point: None,
        }
    }

    pub fn add_node<F, Fut>(&mut self, name: impl Into<String>, func: F) -> &mut Self
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = NodeResult<T>> + Send + 'static,
    {
        let name = name.into();
        let boxed_fn: NodeFn<T> = Box::new(move |input| Box::pin(func(input)));
        self.nodes.insert(name, boxed_fn);
        self
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) -> &mut Self {
        self.edges.push(Edge {
            from: from.into(),
            to: to.into(),
        });
        self
    }

    pub fn add_conditional_edges<F>(
        &mut self,
        from: impl Into<String>,
        condition: F,
        mapping: HashMap<String, String>,
    ) -> &mut Self
    where
        F: Fn(&T) -> String + Send + Sync + 'static,
    {
        self.conditional_edges.push(ConditionalEdge {
            from: from.into(),
            condition: Box::new(condition),
            mapping,
        });
        self
    }

    pub fn set_entry_point(&mut self, node: impl Into<String>) -> &mut Self {
        let node_name = node.into();
        self.edges.push(Edge {
            from: START.to_string(),
            to: node_name.clone(),
        });
        self.entry_point = Some(node_name);
        self
    }

    pub fn set_finish_point(&mut self, node: impl Into<String>) -> &mut Self {
        let node_name = node.into();
        self.edges.push(Edge {
            from: node_name.clone(),
            to: END.to_string(),
        });
        self.finish_point = Some(node_name);
        self
    }

    pub fn compile(self) -> Result<CompiledGraph<T>, GraphError> {
        self.validate()?;
        Ok(CompiledGraph {
            nodes: self.nodes,
            edges: self.edges,
            conditional_edges: self.conditional_edges,
            entry_point: self.entry_point,
        })
    }

    fn validate(&self) -> Result<(), GraphError> {
        for edge in &self.edges {
            if edge.from != START && !self.nodes.contains_key(&edge.from) {
                return Err(GraphError::NodeNotFound(edge.from.clone()));
            }
            if edge.to != END && !self.nodes.contains_key(&edge.to) {
                return Err(GraphError::NodeNotFound(edge.to.clone()));
            }
        }

        for conditional_edge in &self.conditional_edges {
            if !self.nodes.contains_key(&conditional_edge.from) {
                return Err(GraphError::NodeNotFound(conditional_edge.from.clone()));
            }
            for target in conditional_edge.mapping.values() {
                if target != END && !self.nodes.contains_key(target) {
                    return Err(GraphError::NodeNotFound(target.clone()));
                }
            }
        }

        Ok(())
    }
}

pub struct CompiledGraph<T> {
    nodes: HashMap<String, NodeFn<T>>,
    edges: Vec<Edge>,
    conditional_edges: Vec<ConditionalEdge<T>>,
    entry_point: Option<String>,
}

impl<T> CompiledGraph<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub async fn execute(&self, input: T) -> Result<T, GraphError> {
        let start_node = self.entry_point.as_ref()
            .ok_or(GraphError::NoEntryPoint)?;
        
        let mut current_data = input;
        let mut current_node = start_node.clone();

        loop {
            if let Some(node_fn) = self.nodes.get(&current_node) {
                current_data = node_fn(current_data).await
                    .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            } else {
                return Err(GraphError::NodeNotFound(current_node));
            }

            let next_node = self.get_next_node(&current_node, &current_data)?;
            if let Some(next) = next_node {
                if next == END {
                    break;
                }
                current_node = next;
            } else {
                break;
            }
        }

        Ok(current_data)
    }

    pub async fn execute_with_start(&self, start_node: &str, input: T) -> Result<T, GraphError> {
        let mut current_data = input;
        let mut current_node = start_node.to_string();

        loop {
            if let Some(node_fn) = self.nodes.get(&current_node) {
                current_data = node_fn(current_data).await
                    .map_err(|e| GraphError::ExecutionError(e.to_string()))?;
            } else {
                return Err(GraphError::NodeNotFound(current_node));
            }

            let next_node = self.get_next_node(&current_node, &current_data)?;
            if let Some(next) = next_node {
                if next == END {
                    break;
                }
                current_node = next;
            } else {
                break;
            }
        }

        Ok(current_data)
    }

    fn get_next_node(&self, current: &str, data: &T) -> Result<Option<String>, GraphError> {
        for conditional_edge in &self.conditional_edges {
            if conditional_edge.from == current {
                let condition_result = (conditional_edge.condition)(data);
                if let Some(target) = conditional_edge.mapping.get(&condition_result) {
                    return Ok(Some(target.clone()));
                }
            }
        }

        for edge in &self.edges {
            if edge.from == current {
                return Ok(Some(edge.to.clone()));
            }
        }

        Ok(None)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("No entry point set. Use set_entry_point() or execute_with_start()")]
    NoEntryPoint,
}