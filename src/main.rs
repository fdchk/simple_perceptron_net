use std::{thread::sleep, time::Duration};

use plotters::{prelude::*};
use rand::Rng;


#[derive(Clone, Copy)]
enum Activation {
    Sigmoid,
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
    fn derivative(&self, output: f64) -> f64 {
        match self {
            Activation::Sigmoid => output * (1.0-output),
        }
    }
}

struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    act_fn: Activation,
}

impl Perceptron{
    fn new(size: usize, act_fn: Activation) -> Self {

        let mut rng = rand::rng();

        let weights: Vec<f64> = (0..size)
            .map(|_| rng.random_range(-1.0..1.0)) // random floats between -1 and 1
            .collect();

        Perceptron { 
                weights, 
                bias: 0.5, 
                act_fn,
            }
    }

    fn forward(&self, inputs: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }
        sum = sum + self.bias;
        self.act_fn.apply(sum)
    }

    fn learn(&mut self, inputs: &[f64], delta: f64, learning_rate: f64) {
        for i in 0..self.weights.len() {
            self.weights[i] += learning_rate*delta*inputs[i];
        }
        self.bias += learning_rate*delta;
    }
}

struct Layer {
    neurons: Vec<Perceptron>,
}

impl Layer {
    fn new(size: usize, input_size: usize, act_fn: Activation) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..size {
            neurons.push(Perceptron::new(input_size, act_fn));
        }
        Layer { neurons }
    }

    fn forward(&self, inputs: &[f64]) -> Vec<f64> {

        //short
        // self.neurons.iter().map(|n| n.forward(inputs)).collect()
        //long
        let mut out:Vec<f64> = Vec::new();
        for neuron in &self.neurons{
            let res = neuron.forward(inputs);
            out.push(res);
        }
        out
    }
}

struct Network {
    layers: Vec<Layer>,
    learning_rate: f64,
}

impl Network {
    fn new(learning_rate: f64) -> Self {
        Network { layers: Vec::new(), learning_rate }
    }

    fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    fn forward(&self, mut inputs: Vec<f64>) -> Vec<f64> {
        for layer in &self.layers {
            inputs = layer.forward(&inputs);
        }
        inputs
    }

    fn loss(&self, data: &Vec<(Vec<f64>, Vec<f64>)>) -> f64 {
        let mut sum = 0.0;
        for (inputs, targets) in data {
            let outputs = self.forward(inputs.clone());
            for (o, t) in outputs.iter().zip(targets.iter()) {
                sum += (t - o).powi(2);
            }
        }
        sum / (data.len() as f64)   
    }

    fn train(&mut self, inputs: &[f64], targets: &[f64]) {
        // Forward pass
        let mut layer_inputs: Vec<Vec<f64>> = Vec::new();
        let mut current_inputs = inputs.to_vec();
        layer_inputs.push(current_inputs.clone());
        let mut layer_outputs: Vec<Vec<f64>> = Vec::new();

        for layer in &self.layers {
            let outputs = layer.forward(&current_inputs);
            layer_outputs.push(outputs.clone());
            current_inputs = outputs.clone();
            layer_inputs.push(current_inputs.clone());
        }

        // Backward pass
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        for (i, layer) in self.layers.iter().rev().enumerate() {
            let mut layer_delta = Vec::new();
            if i == 0 {
                // output layer
                let outputs = &layer_outputs[layer_outputs.len() - 1];
                for (o, t) in outputs.iter().zip(targets.iter()) {
                    let delta = (t - o) * Activation::Sigmoid.derivative(*o);
                    layer_delta.push(delta);
                }
            } else {
                // hidden layer
                let next_deltas = &deltas[0];
                let next_layer = &self.layers[self.layers.len() - i];
                for j in 0..layer.neurons.len() {
                    let mut sum = 0.0;
                    for k in 0..next_layer.neurons.len() {
                        sum += next_layer.neurons[k].weights[j] * next_deltas[k];
                    }
                    let output = layer_outputs[layer_outputs.len() - i - 1][j];
                    layer_delta.push(sum * Activation::Sigmoid.derivative(output));
                }
            }
            deltas.insert(0, layer_delta);
        }

        // Update weights
        for (l_idx, layer) in self.layers.iter_mut().enumerate() {
            let inputs = &layer_inputs[l_idx];
            for (n_idx, neuron) in layer.neurons.iter_mut().enumerate() {
                neuron.learn(inputs, deltas[l_idx][n_idx], self.learning_rate);
            }
        }
    }
}

fn color_from_value(v: f64) -> RGBColor {
    let v = v.clamp(0.0, 1.0);
    let r = (v * 255.0) as u8;
    let g = 128u8;
    let b = ((1.0 - v) * 255.0) as u8;
    RGBColor(r, g, b)
}

fn plot_decision(net: &Network) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("decision.png", (400, 400)).into_drawing_area();
    root.fill(&RGBColor(50u8,50u8,50u8))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Decision Plot", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;

    chart.configure_mesh().draw()?;

    // plot a grid
    let steps = 20; // resolution
    for i in 0..=steps {
        for j in 0..=steps {
            let x = i as f64 / steps as f64;
            let y = j as f64 / steps as f64;
            let out = net.forward(vec![x, y])[0];
            let color = color_from_value(out); // red = 1, blue = 0
            chart.draw_series(std::iter::once(Circle::new((x, y), 5, color.filled())))?;
        }
    }

    root.present()?;
    println!("Decision plot saved to decision.png");
    Ok(())
}


fn main() {
    let mut net = Network::new(0.5);
    // net.add_layer(Layer::new(2, 2, Activation::Sigmoid)); // output layer: 1 neuron
    net.add_layer(Layer::new(1, 2, Activation::Sigmoid)); // output layer: 1 neuron

        // matrix example
    // let training_data = vec![
    //     (vec![0.0, 0.0],  vec![1.0]), (vec![0.0, 0.25],  vec![0.0]), (vec![0.0, 0.5],  vec![1.0]), (vec![0.0, 0.75],  vec![1.0]), (vec![0.0, 1.0],  vec![1.0]),
    //     (vec![0.25, 0.0], vec![1.0]), (vec![0.25, 0.25], vec![0.0]), (vec![0.25, 0.5], vec![1.0]), (vec![0.25, 0.75], vec![0.0]), (vec![0.25, 1.0], vec![1.0]),
    //     (vec![0.5, 0.0],  vec![1.0]), (vec![0.5, 0.25],  vec![0.0]), (vec![0.5, 0.5],  vec![1.0]), (vec![0.5, 0.75],  vec![0.0]), (vec![0.5, 1.0],  vec![1.0]),
    //     (vec![0.75, 0.0], vec![1.0]), (vec![0.75, 0.25], vec![0.0]), (vec![0.75, 0.5], vec![0.0]), (vec![0.75, 0.75], vec![0.0]), (vec![0.75, 1.0], vec![1.0]),
    //     (vec![1.0, 0.0],  vec![1.0]), (vec![1.0, 0.25],  vec![1.0]), (vec![1.0, 0.5],  vec![1.0]), (vec![1.0, 0.75],  vec![1.0]), (vec![1.0, 1.0],  vec![1.0]),
    // ];

        // XOR example
    let training_data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.5]),
        (vec![1.0, 0.0], vec![0.5]),
        (vec![1.0, 1.0], vec![1.0]),
    ];


    for epoch in 0..100000 {
        for (inputs, targets) in &training_data {
            net.train(inputs, targets);
        }
        if epoch % 1000 == 0 {
            println!("epoch {}", epoch.to_string());
            println!("loss {}", net.loss(&training_data).to_string());
            // plot_decision(&net).unwrap();
            // sleep(Duration::from_millis(500));
        }
    }

    println!("Predictions after training:");
    for (inputs, _) in &training_data {
        let output = net.forward(inputs.clone());
        println!("{:?} -> {:?}", inputs, output);
    }
    println!("mapped predictions:");
    let zero_range = 0.33..0.66;
    let mut mapped_output;
    for (inputs, _) in &training_data {
        let output = net.forward(inputs.clone())[0];
        if zero_range.contains(&output) {
            mapped_output = 1;
        } else {
            mapped_output = 0;
        }
        println!("{:?} -> {:?}", inputs, mapped_output);
    }
plot_decision(&net).unwrap();


}