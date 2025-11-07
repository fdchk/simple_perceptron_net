use std::{fs::File, io::{self, Write}, iter::StepBy, time::Instant};
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

struct Architecture {
    layer_count: u32,
    input_params_count: u32,
    hidden_neurons_count: Vec<u32>,
    hidden_layers_count: u32,
    output_neurons_count: u32,
    learning_rate: f64,
    epochs: u64
}

impl Architecture {
    fn init() -> Self {
        let layer_count: u32;
        let input_params_count: u32;
        let mut hidden_neurons_count: Vec<u32> = Vec::new();
        let output_neurons_count: u32;
        let learning_rate: f64;
        let epochs: u64;
        println!("Network arch wizard");
        println!("--------------------");
        input_params_count = query_user("Enter number of input parameters: (bundled data requires 2, asserted for now)") as u32;
        assert_eq!(input_params_count, 2);
        layer_count = query_user("Total layer count: (2 is sufficient)") as u32;
        let hidden_layers_count = layer_count - 1;
        for i in 0..hidden_layers_count {
            let question = format!("Neurons on layer {}: ", i+1);
            hidden_neurons_count.push(query_user(&question) as u32);
        }
        output_neurons_count = query_user("Output count: (bundled data requires 1, asserted for now)") as u32;
        assert_eq!(output_neurons_count, 1);
        epochs = query_user("Train for how many epochs: (go ham, limit is u64)") as u64;
        learning_rate = query_user("Learning rate: (value is in f64 so mind the format");


        Architecture{
            layer_count,
            input_params_count,
            hidden_neurons_count,
            hidden_layers_count,
            output_neurons_count,
            learning_rate,
            epochs
        }
    }

    fn print(&self) {
        println!("Network arch:");
        println!("------------------");
        println!("Layers:                       {}", self.layer_count);
        println!("Hidden layers:                {}", self.hidden_layers_count);
        println!("Input params:                 {}", self.input_params_count);
        for i in 0..self.hidden_layers_count {
            println!("Neurons on layer {}:      {}", i+1, self.hidden_neurons_count[i as usize]);
        }
        println!("Output neurons:               {}", self.output_neurons_count);
        println!("Epoch cap:                    {}", self.epochs);
        println!("Learning rate:                {}", self.learning_rate);
        println!("------------------");
    }
}

fn query_user(question: &str) -> f64 {
    let mut buffer: String = String::new();
    println!("{}", question);
    io::stdin().read_line(&mut buffer).expect("Failed to read input");
    let output = buffer.trim().parse::<f64>().expect("Failed to parse input");
    output
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

    fn print(&self) {
        println!("Network actual conf:");
        println!("------------------");
        println!("Layers:");
        println!("------------------");
        for (i, layer) in self.layers.iter().enumerate() {
            println!("  Layer {}", i);
            println!("  Neurons: {}", layer.neurons.len().to_string());
            let params: usize = layer.neurons.iter().map(|n| n.weights.len()).sum();
            println!("  Trainable params: {}", params.to_string());
            println!("------------------");
        }
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

fn plot_decision(net: &Network, epoch: i32) -> Result<(), Box<dyn std::error::Error>> {
    let filename = format!("decision_{}.png", epoch);
    let root = BitMapBackend::new(&filename, (400, 400)).into_drawing_area();
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
    println!("Decision plot saved to {}", &filename);
    Ok(())
}

fn main() {
    let arch = Architecture::init();
    let mut net = Network::new(arch.learning_rate);

    net.add_layer(Layer::new(arch.hidden_neurons_count[0 as usize] as usize, arch.input_params_count as usize, Activation::Sigmoid)); // layer 1, input params -> hidden_neur
    for layer in 0..arch.hidden_layers_count-1 {
        net.add_layer(Layer::new(arch.hidden_neurons_count[(layer+1) as usize] as usize, arch.hidden_neurons_count[layer as usize] as usize, Activation::Sigmoid));
    }
    net.add_layer(Layer::new(arch.output_neurons_count as usize, *arch.hidden_neurons_count.last().expect("array malfunction") as usize, Activation::Sigmoid)); // layer n, hidden_neur -> output params

    arch.print();
    net.print();


    let matrix = vec![
        (vec![0.0, 0.0],  vec![0.0]), (vec![0.0, 0.25],  vec![1.0]), (vec![0.0, 0.5],  vec![0.0]), (vec![0.0, 0.75],  vec![1.0]), (vec![0.0, 1.0],  vec![0.0]),
        (vec![0.25, 0.0], vec![1.0]), (vec![0.25, 0.25], vec![0.0]), (vec![0.25, 0.5], vec![0.0]), (vec![0.25, 0.75], vec![0.0]), (vec![0.25, 1.0], vec![1.0]),
        (vec![0.5, 0.0],  vec![0.0]), (vec![0.5, 0.25],  vec![0.0]), (vec![0.5, 0.5],  vec![1.0]), (vec![0.5, 0.75],  vec![0.0]), (vec![0.5, 1.0],  vec![0.0]),
        (vec![0.75, 0.0], vec![1.0]), (vec![0.75, 0.25], vec![0.0]), (vec![0.75, 0.5], vec![0.0]), (vec![0.75, 0.75], vec![0.0]), (vec![0.75, 1.0], vec![1.0]),
        (vec![1.0, 0.0],  vec![0.0]), (vec![1.0, 0.25],  vec![1.0]), (vec![1.0, 0.5],  vec![0.0]), (vec![1.0, 0.75],  vec![1.0]), (vec![1.0, 1.0],  vec![0.0]),
    ];

        // XOR example
    let xor = vec![
        (vec![0.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];

    println!("decide on dataset: ");
    println!("   1: 5x5 matrix");
    println!("   2: XOR");

    let mut buffer: String = String::new();
    io::stdin().read_line(&mut buffer).expect("Failed to read input");
    let output = buffer.trim().parse::<u8>().expect("Invalid input");

    let training_data;
    match output {
        1 => {
            training_data = matrix;
        }
        2 => {
            training_data = xor;
        }
        _ => {
            panic!("no training data selected");
        }
    }
    // matrix example

    let mut csv: File = File::create("stats.csv").expect("file create failed");
    let mut buf;

    let mut timer = Instant::now();
    let mut prev_epoch= 0;
    for epoch in 0..arch.epochs {
        // let epoch_start = Instant::now();
        for (inputs, targets) in &training_data {
            net.train(inputs, targets);
        }
        let loss = net.loss(&training_data);
        let mut step_div:String = arch.epochs.to_string();
        step_div.truncate(4);
        let step = step_div.parse::<u64>().expect("wrong parse");
        if epoch % step == 0 {
            println!("epoch {} of {} | {} done in {:.2} | loss {:.8}", epoch, arch.epochs, epoch - prev_epoch, timer.elapsed().as_secs_f64(), loss);
            prev_epoch = epoch;
            buf = format!("{};{}\n", &epoch, &loss);
            let towrite = buf.as_bytes();
            csv.write(towrite).expect("file write error");
            timer = Instant::now();
        }
        if loss <= 0.00001 {
            println!("loss is sufficiently low ({})", loss);
            break
        }
    }

    println!("Predictions after training:");
    for (inputs, _) in &training_data {
        let output = net.forward(inputs.clone());
        println!("{:?} -> {:?}", inputs, output);
    }
    // println!("mapped predictions:");
    // let zero_range = 0.33..0.66;
    // let mut mapped_output;
    // for (inputs, _) in &training_data {
    //     let output = net.forward(inputs.clone())[0];
    //     if zero_range.contains(&output) {
    //         mapped_output = 1;
    //     } else {
    //         mapped_output = 0;
    //     }
    //     println!("{:?} -> {:?}", inputs, mapped_output);
    // }
plot_decision(&net, 0).unwrap();


}