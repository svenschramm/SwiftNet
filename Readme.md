# SwiftNet

SwiftNet is a simple and powerful neural network framework for Swift. 

## Tutorial

In this tutorial we are going to create and train a two layer fully-connected neural network with SwiftNet.

Before we will write any code let's think about the operations that are necessary to compute the output of the network. We can divide this task into four steps. The first step would be to take the input of the neural network and the weights and bias of the first layer to compute the net output of the first layer. We do this by calling the function `fullyConnected` with `input`, `weights1` and `bias1` as arguments where `input` is the input of the neural network and `weights1` and `bias1` are the weights and bias of the first layer:

```swift
output = fullyConnected(input, weights1, bias1)
```

Now `output` is the net output of the first layer. To compute the final output of the first layer we would pass the output to an activation function: 

```swift
output = activation(output)
```

To compute the output of the second fully-conntected layer we would repeat step 1, with arguments `output`, `weights2` and `bias2` where `output` is the output of the first fully-connected layer and `weights2` and `bias2` are the weights and bias of the second layer:

```swift
output = fullyConnected(output, weights2, bias2)
```

If we want to do multiclass classification we would have to feed the output to the softmax function which would give us the final output of our neural network:

```swift
output = softmax(output)
```

The steps required to construct a neural network in SwiftNet are almost identical to the steps above. There is only one big difference. The functions we are going to use don't compute the output of the network directly. Instead SwiftNet is going to create a computational graph that can not only be used to compute the output but is also used to train the network. Here is what the code looks like:

```swift
let input = Node.variable(name: "input")
let labels = Node.variable(name: "labels")
let output = Node.fullyConnected(input, units: 128)
output = Node.activation(output, activationFunction: .rectifiedLinear)
output = Node.fullyConnected(output, units: 10)
output = Node.softmaxOutput(output, labels)
```

To train the network you need to create a Network object:

```swift
let network = Network(outputNodes: [output])
```

Training the neural network is just one line of code:

```swift
network.fit(data: trainingData, batchSize: 10, learningRate: 0.1, epochs: 3)
```

A complete example that shows how to train a neural network for handwriting recognition can be found in the examples dictionary of this repository.

## Getting Started

1. Drag and drop the SwiftNet.xcodeproj into your Xcode project
2. Under "Build Phases" -> "Link Binary With Libraries" click on "+" and select "SwiftNet.framework"
3. Add `import SwiftNet` to files that use the SwiftNet framework 
