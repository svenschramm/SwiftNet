import XCTest
import SwiftNet

class SwiftNetTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
    }
    
    override func tearDown() {
        super.tearDown()
    }
    
    func testConvolutionWithMultipleInputChannels() {
        let node = Node.convolution(Node.variable(name: "input"), kernelWidth: 2, kernelHeight: 2, filters: 1)
        
        let data = NDArray(shape: [2, 4, 4], elements: [0.1, 0.2, 0.3, 0.4,
                                                        0.2, 0.3, 0.4, 0.5,
                                                        0.3, 0.4, 0.5, 0.6,
                                                        0.4, 0.5, 0.6, 0.7,
                                                        0.1, 0.2, 0.3, 0.4,
                                                        0.2, 0.3, 0.4, 0.5,
                                                        0.3, 0.4, 0.5, 0.6,
                                                        0.4, 0.5, 0.6, 0.7])
        let weights = NDArray(shape: [1, 2, 2, 2], elements: [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
        let bias = NDArray([0.5])
        
        let input = [data, weights, bias]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.96, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 1.16, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 1.36, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[3], 1.16, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[4], 1.36, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[5], 1.56, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[6], 1.36, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[7], 1.56, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[8], 1.76, accuracy: 0.00001)

        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[2], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[3], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[4], 0.4, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[5], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[6], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[7], 0.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[8], 0.4, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[9], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[10], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[11], 0.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[12], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[13], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[14], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[15], 0.4, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[16], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[17], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[18], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[19], 0.2, accuracy: 0.00001)
    
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 2.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[2], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[3], 4.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[4], 2.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[5], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[6], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[7], 4.5, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[0], 9.0, accuracy: 0.00001)
    }
    
    func testConvolutionWithMultipleOutputChannels() {
        let node = Node.convolution(Node.variable(name: "input"), kernelWidth: 2, kernelHeight: 2, filters: 2)
        
        let data = NDArray(shape: [1, 4, 4], elements: [0.1, 0.2, 0.3, 0.4,
                                                        0.2, 0.3, 0.4, 0.5,
                                                        0.3, 0.4, 0.5, 0.6,
                                                        0.4, 0.5, 0.6, 0.7])
        let weights = NDArray(shape: [2, 1, 2, 2], elements: [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
        let bias = NDArray([0.5, 0.5])
        
        let input = [data, weights, bias]
        let output = node.op.forward(input: input)

        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.73, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.83, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 0.93, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[9], 0.73, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[10], 0.83, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[11], 0.93, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[2], 0.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[3], 0.4, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[4], 0.8, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[5], 2.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[6], 2.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[7], 1.2, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 2.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[2], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[3], 4.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[4], 2.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[5], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[6], 3.6, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[7], 4.5, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[0], 9.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[1], 9.0, accuracy: 0.00001)
    }
    
    func testPooling() {
        let node = Node.pooling(Node.variable(name: "input"), poolingFunction: .max, kernelWidth: 2, kernelHeight: 2, xStride: 2, yStride: 2)
        
        let input = [NDArray(shape: [1, 4, 4], elements: [0.15, 0.1, 0.2, 0.25, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.35, 0.3, 0.4, 0.45])]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)

        XCTAssertEqualWithAccuracy(output.elements[0], 0.15, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.25, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 0.35, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[3], 0.45, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[2], 0.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[3], 1.0, accuracy: 0.00001)
    }
    
    func testLinearRegressionOutput() {
        let node = Node.linearRegressionOutput(Node.variable(name: "input"), Node.variable(name: "target"))
        
        let data = NDArray(shape: [1], elements: [0.5])
        let labels = NDArray(shape: [1], elements: [0.6])
        
        let input = [data, labels]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], -0.1, accuracy: 0.00001)
    }
    
    func testLogisticRegressionOutput() {
        let node = Node.logisticRegressionOutput(Node.variable(name: "input"), Node.variable(name: "labels"))

        let data = NDArray(shape: [1], elements: [0.5])
        let labels = NDArray(shape: [1], elements: [0.6])
        
        let input = [data, labels]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.62245, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.02245, accuracy: 0.00001)
    }
    
    func testSoftmaxOutput() {
        let node = Node.softmaxOutput(Node.variable(name: "input"), Node.variable(name: "labels"))
        
        let data = NDArray(shape: [2], elements: [0.5, 0.6])
        let labels = NDArray(shape: [2], elements: [0.2, 0.3])
        
        let input = [data, labels]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.47502, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.52497, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.27502, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.22497, accuracy: 0.00001)
    }
    
    
    func testSigmoidActivation() {
        let node = Node.activation(Node.variable(name: "input"), activationFunction: .sigmoid)
        
        let data = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [data]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.52497, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.54983, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.24937, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.24751, accuracy: 0.00001)
    }
    
    func testTanhActivation() {
        let node = Node.activation(Node.variable(name: "input"), activationFunction: .tanh)
        
        let data = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [data]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.11422, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.22744, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.13886, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.12384, accuracy: 0.00001)
    }
    
    func testRectifiedLinearActivation() {
        let node = Node.activation(Node.variable(name: "input"), activationFunction: .rectifiedLinear)
        
        let data = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [data]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
    }
    
    func testFullyConnected() {
        let node = Node.fullyConnected(Node.variable(name: "input"), units: 3)
        
        let data = NDArray(shape: [2], elements: [0.1, 0.2])
        let weights = NDArray(shape: [3, 2], elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        let bias = NDArray([0.1, 0.2, 0.3])
        
        let input = [data, weights, bias]
        let output = node.op.forward(input: input)

        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.15, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.31, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 0.47, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.9, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[2], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[3], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[4], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[5], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[2]!.elements[2], 1.0, accuracy: 0.00001)
    }
    
    func testJoin() {
        let node = Node.join([Node.variable(name: "data1"), Node.variable(name: "data2")])
        
        let data1 = NDArray(shape: [2], elements: [0.1, 0.2])
        let data2 = NDArray(shape: [2], elements: [0.3, 0.4])
        
        let input = [data1, data2]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[3], 0.4, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 1.0, accuracy: 0.00001)
    }
    
    func testReshape() {
        let node = Node.reshape(Node.variable(name: "data"), shape: [1, 2])
        
        let data = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [data]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.2, accuracy: 0.00001)
        XCTAssertEqual(output.shape, [1, 2])
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqual(gradients[0]!.shape, [2])
    }
    
    func testFlatten() {
        let node = Node.flatten(Node.variable(name: "data"))
        
        let data = NDArray(shape: [1, 2], elements: [0.1, 0.2])
        
        let input = [data]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.2, accuracy: 0.00001)
        XCTAssertEqual(output.shape, [2])
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqual(gradients[0]!.shape, [1, 2])
    }
    
    func testElementWiseInferShapes() {
        let node = Node.add(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], [2]]).output, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], nil]).input[0]!, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], nil]).input[1]!, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], nil]).output, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [nil, [2]]).input[0]!, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [nil, [2]]).input[1]!, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [nil, [2]]).output, [2])
    }
    
    func testAdd() {
        let node = Node.add(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.2, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.4, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 1.0, accuracy: 0.00001)
    }
    
    func testSubtract() {
        let node = Node.subtract(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2], elements: [0.1, 0.2])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.0, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], -1.0, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], -1.0, accuracy: 0.00001)
    }
    
    func testMultiply() {
        let node = Node.multiply(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2], elements: [0.3, 0.4])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.03, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.08, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.4, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 0.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 0.2, accuracy: 0.00001)
    }
    
    func testDivide() {
        let node = Node.divide(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2], elements: [0.3, 0.4])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.33333, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.5, accuracy: 0.00001)
        
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 3.33333, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 2.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], -1.11111, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], -1.25, accuracy: 0.00001)
    }
    
    func testDotInferShapes() {
        let node = Node.dot(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2, 3], [3, 2]]).output, [2, 2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2, 3], [3]]).output, [2])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], [2, 3]]).output, [3])
        XCTAssertEqual(node.op.inferShapes(inShapes: [[2], [2]]).output, [1])
    }
    
    func testMatrixMatrixMultiplication() {
        let node = Node.dot(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2, 3], elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        let right = NDArray(shape: [3, 2], elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.22, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[1], 0.28, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[2], 0.49, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(output.elements[3], 0.64, accuracy: 0.00001)
    
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[0], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[1], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[2], 1.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[3], 0.3, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[4], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[0]!.elements[5], 1.1, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[0], 0.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[1], 0.5, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[2], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[3], 0.7, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[4], 0.9, accuracy: 0.00001)
        XCTAssertEqualWithAccuracy(gradients[1]!.elements[5], 0.9, accuracy: 0.00001)
    }

    func testMatrixVectorMultiplication() {
        let node = Node.dot(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2, 3], elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        let right = NDArray(shape: [3], elements: [0.1, 0.2, 0.3])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqual(output.shape, [2])
        XCTAssertEqual(gradients[0]!.shape, [2, 3])
        XCTAssertEqual(gradients[1]!.shape, [3])
    }
    
    func testVectorMatrixMultiplication() {
        let node = Node.dot(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2, 3], elements: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqual(output.shape, [3])
        XCTAssertEqual(gradients[0]!.shape, [2])
        XCTAssertEqual(gradients[1]!.shape, [2, 3])
    }
    
    func testVectorVectorMultiplication() {
        let node = Node.dot(Node.variable(name: "lhs"), Node.variable(name: "rhs"))
        
        let left = NDArray(shape: [2], elements: [0.1, 0.2])
        let right = NDArray(shape: [2], elements: [0.3, 0.4])
        
        let input = [left, right]
        let output = node.op.forward(input: input)
        
        let gradients = node.op.backward(delta: NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size)), input: input, output: output)
        
        XCTAssertEqualWithAccuracy(output.elements[0], 0.11, accuracy: 0.00001)
        XCTAssertEqual(output.shape, [1])
        XCTAssertEqual(gradients[0]!.shape, [2])
        XCTAssertEqual(gradients[1]!.shape, [2])
    }
    
    func testFullyConnectedNetwork() {
        let fullyConnected1 = Node.activation(Node.fullyConnected(Node.variable(name: "input"), units: 2), activationFunction: .sigmoid)
        let fullyConnected2 = Node.activation(Node.fullyConnected(fullyConnected1, units: 2), activationFunction: .sigmoid)
        let linearRegressionOutput = Node.linearRegressionOutput(fullyConnected2, Node.variable(name: "target"))
        let network = Network(outputNodes: [linearRegressionOutput])
        
        let input = NDArray(shape: [2], elements: [0.1, 0.2])
        let target = NDArray(shape: [2], elements: [0.5, 0.6])

        let data = ["input": [input], "target": [target]]
        
        network.fit(data: data, batchSize: 1, learningRate: 0.5, epochs: 100)
        let output = network.predict(data: ["input": [input]])
        
        XCTAssertEqualWithAccuracy(output.first![0].elements[0], 0.5, accuracy: 0.01)
        XCTAssertEqualWithAccuracy(output.first![0].elements[1], 0.6, accuracy: 0.01)
    }
}
