import Foundation
import Accelerate

public typealias Shape = [Int]


func shuffle(_ dict: inout [String: [NDArray]]) {
    precondition(dict.values.count > 0)
    let count = dict.values.first!.count
    if count < 2 {
        return
    }
    var keys = [String]()
    var arrays = [[NDArray]]()
    for (key, array) in dict {
        keys.append(key)
        arrays.append(array)
    }
    for i in 0..<count - 1 {
        let j = Int(arc4random_uniform(UInt32(count - i))) + i
        guard i != j else { continue }
    
        for k in 0..<arrays.count {
            swap(&arrays[k][i], &arrays[k][j])
        }
    }
    
    for (key, array) in zip(keys, arrays) {
        dict[key] = array
    }
}

public func randn() -> Double {
    let u = drand48()
    let v = drand48()
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v)
}

public func pack(_ array: [Float32]) -> Data {
    return Data(buffer: UnsafeBufferPointer(start: array, count: array.count))
}

public func unpack(_ data: Data) -> [Float32] {
    return data.withUnsafeBytes {
        Array(UnsafeBufferPointer<Float32>(start: $0, count: data.count / MemoryLayout<Float32>.size))
    }
}

public func sigmoid(_ z: Float32) -> Float32 {
    return 1.0 / (1.0 + exp(-z))
}

public func add(_ a: NDArray, _ b: NDArray) -> NDArray {
    precondition(a.shape == b.shape)
    var elements = b.elements
    cblas_saxpy(Int32(a.size), 1.0, UnsafePointer<Float32>(a.elements), 1, &(elements), 1)
    return NDArray(shape: a.shape, elements: elements)
}

public func subtract(_ a: NDArray, _ b: NDArray) -> NDArray {
    precondition(a.shape == b.shape)
    var elements = a.elements
    cblas_saxpy(Int32(a.size), -1.0, UnsafePointer<Float32>(b.elements), 1, &(elements), 1)
    return NDArray(shape: a.shape, elements: elements)
}

// multiply vector with scalar
public func multiply(_ a: NDArray, _ c: Float32) -> NDArray {
    var elements = a.elements
    cblas_sscal(Int32(a.size), c, &(elements), 1)
    return NDArray(shape: a.shape, elements: elements)
}

// hadamard product
public func multiply(_ a: NDArray, _ b: NDArray) -> NDArray {
    precondition(a.shape == b.shape)
    let shape = a.shape
    let size = a.size
    let a = a.elements
    let b = b.elements
    var elements = [Float32](repeating: 0.0, count: size)
    for i in 0..<size {
        elements[i] = a[i] * b[i]
    }
    return NDArray(shape: shape, elements: elements)
}

public func divide(_ a: NDArray, _ b: NDArray) -> NDArray {
    precondition(a.shape == b.shape)
    let shape = a.shape
    let size = a.size
    let a = a.elements
    let b = b.elements
    var elements = [Float32](repeating: 0.0, count: size)
    for i in 0..<size {
        elements[i] = a[i] / b[i]
    }
    return NDArray(shape: shape, elements: elements)
}

// multiply column vector with row vector
public func outer(_ x: NDArray, _ y: NDArray) -> NDArray {
    precondition(x.shape.count == 1 && y.shape.count == 1)
    var elements = [Float32](repeating: 0, count: x.size * y.size)
    cblas_sger(CblasRowMajor, Int32(x.size), Int32(y.size), 1.0, UnsafePointer<Float32>(x.elements), 1, UnsafePointer<Float32>(y.elements), 1, &(elements), Int32(y.size))
    return NDArray(shape: [x.size, y.size], elements: elements)
}

public func dot(_ a: NDArray, _ b: NDArray) -> NDArray {
    if a.shape.count == 2 && b.shape.count == 2 {
        return gemm(a, b)
    } else if a.shape.count == 2 && b.shape.count == 1 {
        return gemv(a, b)
    } else if a.shape.count == 1 && b.shape.count == 2 {
        return gemv(b, a, transposeA: true)
    } else if a.shape.count == 1 && b.shape.count == 1 {
        let n = a.shape[0]
        let result = cblas_sdot(Int32(n), UnsafePointer<Float32>(a.elements), 1, UnsafePointer<Float32>(b.elements), 1)
        return NDArray([result])
    }
    fatalError()
}

public func gemm(_ a: NDArray, _ b: NDArray, transposeA: Bool = false, transposeB: Bool = false) -> NDArray {
    precondition(a.shape.count == 2 && b.shape.count == 2)
    precondition(a.shape[transposeA ? 0 : 1] == b.shape[transposeB ? 1 : 0])
    
    let m = transposeA ? a.shape[1] : a.shape[0]
    let n = transposeB ? b.shape[0] : b.shape[1]
    let k = transposeA ? a.shape[0] : a.shape[1]
    
    let lda = transposeA ? m : k
    let ldb = transposeB ? k : n
    let ldc = n
    
    var elements = [Float32](repeating: 0, count: m * n)
    cblas_sgemm(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, transposeB ? CblasTrans : CblasNoTrans, Int32(m), Int32(n), Int32(k), 1.0, UnsafePointer<Float32>(a.elements), Int32(lda), UnsafePointer<Float32>(b.elements), Int32(ldb), 0.0, &(elements), Int32(ldc))
    return NDArray(shape: [m, n], elements: elements)
}

public func gemv(_ a: NDArray, _ x: NDArray, transposeA: Bool = false) -> NDArray {
    precondition(a.shape.count == 2 && x.shape.count == 1 && x.shape[0] == a.shape[transposeA ? 0 : 1])
    let m = a.shape[0]
    let n = a.shape[1]
    var elements = [Float32](repeating: 0, count: transposeA ? n : m)
    cblas_sgemv(CblasRowMajor, transposeA ? CblasTrans : CblasNoTrans, Int32(m), Int32(n), 1.0, UnsafePointer<Float32>(a.elements), Int32(n), UnsafePointer<Float32>(x.elements), 1, 1.0, &(elements), 1)
    return NDArray(elements)
}

func round(_ array: NDArray) -> NDArray {
    let shape = array.shape
    let elements = array.elements.map { round($0) }
    return NDArray(shape: shape, elements: elements)
}

func abs(_ array: NDArray) -> NDArray {
    let shape = array.shape
    let elements = array.elements.map { abs($0) }
    return NDArray(shape: shape, elements: elements)
}

func equal(_ a: NDArray, _ b: NDArray) -> NDArray {
    precondition(a.shape == b.shape)
    let shape = a.shape
    let size = a.size
    let a = a.elements
    let b = b.elements
    var elements = [Float32](repeating: 0.0, count: size)
    for i in 0..<size {
        elements[i] = a[i] == b[i] ? 1 : 0
    }
    return NDArray(shape: shape, elements: elements)
}

func mean(_ array: NDArray) -> NDArray {
    let count = array.size
    let sum = array.elements.reduce(0.0, +)
    return NDArray(shape: [1], elements: [sum / Float32(count)])
}

public func xavier(shape: [Int]) -> NDArray  {
    let size = shape.reduce(1, *)
    let fanIn = shape.count == 4 ? shape[1] * shape[2] * shape[3] : shape[1]
    let fanOut = shape.count == 4 ? shape[0] * shape[2] * shape[3] : shape[0]
    let scale = sqrtf(1.0 / (Float32(fanIn + fanOut) / 2))
    var elements = [Float32](repeating: 0.0, count: size)
    for i in 0..<size {
        elements[i] = Float32(randn()) * scale
    }
    return NDArray(shape: shape, elements: elements)
}

public func zero(shape: [Int]) -> NDArray {
    return NDArray(shape: shape, elements: [Float32](repeating: 0.0, count: shape.reduce(1, *)))
}

struct EvaluationMetricFunctions {
    
    struct Binary {

        static func accuracy(_ predicted: [NDArray], _ actual: [NDArray]) -> Float32 {
            var sum: Float32 = 0.0
            var count = 0

            for (actual, predicted) in zip(actual, predicted) {
                sum += mean(equal(round(predicted), actual)).elements.first!
                count += 1
            }

            return sum / Float32(count)
        }

    }
    
    struct Multiclass {
        
        static func accuracy(_ predicted: [NDArray], _ actual: [NDArray]) -> Float32 {
            var sum: Float32 = 0.0
            var count = 0
            
            for (actual, predicted) in zip(actual, predicted) {
                let index1 = predicted.elements.index(of: predicted.elements.max()!)!
                let index2 = actual.elements.index(of: actual.elements.max()!)!
                if index1 == index2 {
                    sum += 1
                }
                count += 1
            }
            
            return sum / Float32(count)
        }
    }
    
    static func meanAbsoluteError(_ predicted: [NDArray], _ actual: [NDArray]) -> Float32 {
        var sum: Float32 = 0.0
        var count = 0
        
        for (actual, predicted) in zip(actual, predicted) {
            sum += mean(abs(subtract(predicted, actual))).elements.first!
            count += 1
        }
        
        return sum / Float32(count)
    }
}

public enum EvaluationMetric {
    case accuracy
    case meanAbsoluteError
}

public class NDArray: NSObject, NSCoding {
    
    public var elements: [Float32]
    public var shape: [Int]
    public var size: Int
    
    public convenience init(_ array: [Float32]) {
        self.init(shape: [array.count], elements: array)
    }
    
    public init(shape: [Int], elements: [Float32]? = nil) {
        self.shape = shape
        self.size = shape.reduce(1, *)
        if elements != nil {
            self.elements = elements!
        } else {
            self.elements = [Float32](repeating: 0, count: self.size)
        }
        super.init()
        
        precondition(self.elements.count == size, "Array shape is invalid")
    }
    
    public required convenience init?(coder decoder: NSCoder) {
        let elements = unpack(decoder.decodeObject(forKey: "elements") as! Data)
        let shape = decoder.decodeObject(forKey: "shape") as! [Int]
        self.init(shape: shape, elements: elements)
    }
    
    public func encode(with coder: NSCoder) {
        coder.encode(pack(elements), forKey: "elements")
        coder.encode(shape, forKey: "shape")
    }
    
    public func reshaped(_ newShape: [Int]) -> NDArray {
        return NDArray(shape: newShape, elements: elements)
    }
}


extension Dictionary where Key: Hashable, Value: ExpressibleByArrayLiteral  {
    
    func reversed() -> [Key: Array<Key>] {
        var reversedMap = [Key: Array<Key>]()
        for key in self.keys {
            reversedMap[key] = []
        }
        
        for (key1, value1) in self {
            for key2 in value1 as! Array<Key> {
                var value2 = reversedMap[key2]
                value2!.append(key1)
                reversedMap[key2] = value2
            }
            
        }
        return reversedMap
    }
}


public func ==(lhs: Node, rhs: Node) -> Bool {
    return lhs === rhs
}


public class Node: NSObject, NSCoding {
    
    public var inputNodes: [Node]
    public var name: String?
    public var attributes: [String: Any]
    
    init(inputNodes: [Node] = [], name: String?, attributes: [String: Any]? = nil) {
        self.inputNodes = inputNodes
        self.name = name
        self.attributes = attributes ?? [String: Any]()
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.inputNodes = decoder.decodeObject(forKey: "inputNodes") as! [Node]
        self.name = decoder.decodeObject(forKey: "name") as? String
        self.attributes = decoder.decodeObject(forKey: "attributes") as! [String: Any]
        super.init()
    }
    
    public func encode(with coder: NSCoder) {
        coder.encode(inputNodes, forKey: "inputNodes")
        coder.encode(name, forKey: "name")
        coder.encode(attributes, forKey: "attributes")
    }
    
    static var nameCounts = [String: Int]()
    
    class func uniqueName(withPrefix prefix: String) -> String {
        var count = nameCounts[prefix] ?? 0
        let name = "\(prefix)\(count)"
        count += 1
        nameCounts[prefix] = count
        return name
    }
    
    public class func add(_ lhs: Node, _ rhs: Node, name: String? = nil) -> OperatorNode {
        let op = AddOperator()
        return OperatorNode(inputNodes: [lhs, rhs], name: name ?? Node.uniqueName(withPrefix: "add"), operator: op)
    }
    
    public class func subtract(_ lhs: Node, _ rhs: Node, name: String? = nil) -> OperatorNode {
        let op = SubtractOperator()
        return OperatorNode(inputNodes: [lhs, rhs], name: name ?? Node.uniqueName(withPrefix: "subtract"), operator: op)
    }
    
    public class func multiply(_ lhs: Node, _ rhs: Node, name: String? = nil) -> OperatorNode {
        let op = MultiplyOperator()
        return OperatorNode(inputNodes: [lhs, rhs], name: name ?? Node.uniqueName(withPrefix: "multiply"), operator: op)
    }
    
    public class func divide(_ lhs: Node, _ rhs: Node, name: String? = nil) -> OperatorNode {
        let op = DivideOperator()
        return OperatorNode(inputNodes: [lhs, rhs], name: name ?? Node.uniqueName(withPrefix: "divide"), operator: op)
    }
    
    public class func dot(_ lhs: Node, _ rhs: Node, name: String? = nil) -> OperatorNode {
        let op = DotOperator()
        return OperatorNode(inputNodes: [lhs, rhs], name: name ?? Node.uniqueName(withPrefix: "dot"), operator: op)
    }
    
    public class func linearRegressionOutput(_ data: Node, _ target: Node?, name: String? = nil) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "linearRegressionOutput")
        let target = target ?? VariableNode(name: "\(name)_target")
        let op = LinearRegressionOutputOperator()
        return OperatorNode(inputNodes: [data, target], name: name, operator: op)
    }
    
    public class func logisticRegressionOutput(_ data: Node, _ labels: Node?, name: String? = nil) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "logisticRegressionOutput")
        let labels = labels ?? VariableNode(name: "\(name)_labels")
        let op = LogisticRegressionOutputOperator()
        return OperatorNode(inputNodes: [data, labels], name: name, operator: op)
    }
    
    public class func softmaxOutput(_ data: Node, _ labels: Node?, name: String? = nil) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "softmaxOutput")
        let labels = labels ?? VariableNode(name: "\(name)_labels")
        let op = SoftmaxOutputOperator()
        return OperatorNode(inputNodes: [data, labels], name: name, operator: op)
    }
    
    public class func reinforce(_ data: Node, _ labels: Node?, _ reward: Node?, name: String? = nil) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "reinforce")
        let labels = labels ?? VariableNode(name: "\(name)_labels")
        let reward = reward ?? VariableNode(name: "\(name)_reward")
        let op = ReinforceOperator()
        return OperatorNode(inputNodes: [data, labels, reward], name: name, operator: op)
    }

    public class func activation(_ data: Node, name: String? = nil, activationFunction: ActivationFunction) -> OperatorNode {
        let op = ActivationOperator(activationFunction: activationFunction)
        return OperatorNode(inputNodes: [data], name: name ?? Node.uniqueName(withPrefix: "activation"), operator: op)
    }
    
    public class func fullyConnected(_ data: Node, _ weights: Node? = nil, _ bias: Node? = nil, name: String? = nil, units: Int) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "fullyConnected")
        let weights = weights ?? VariableNode(name: "\(name)_weights", attributes: ["initializer": "xavier"])
        let bias = bias ?? VariableNode(name: "\(name)_bias", attributes: ["initializer": "zero"])
        let op = FullyConnectedOperator(units: units)
        return OperatorNode(inputNodes: [data, weights, bias], name: name, operator: op)
    }
    
    public class func convolution(_ data: Node, _ weights: Node? = nil, _ bias: Node? = nil, name: String? = nil, kernelWidth: Int, kernelHeight: Int, xStride: Int = 1, yStride: Int = 1, xPadding: Int = 0, yPadding: Int = 0, filters: Int) -> OperatorNode {
        let name = name ?? Node.uniqueName(withPrefix: "convolution")
        let weights = weights ?? VariableNode(name: "\(name)_weights", attributes: ["initializer": "xavier"])
        let bias = bias ?? VariableNode(name: "\(name)_bias", attributes: ["initializer": "zero"])
        let op = ConvolutionOperator(kernelWidth: kernelWidth, kernelHeight: kernelHeight, xStride: xStride, yStride: yStride, xPadding: xPadding, yPadding: yPadding, filters: filters)
        return OperatorNode(inputNodes: [data, weights, bias], name: name, operator: op)
    }
    
    public class func pooling(_ data: Node, name: String? = nil, poolingFunction: PoolingFunction, kernelWidth: Int, kernelHeight: Int, xStride: Int = 1, yStride: Int = 1, xPadding: Int = 0, yPadding: Int = 0) -> OperatorNode {
        let op = PoolingOperator(poolingFunction: poolingFunction, kernelWidth: kernelWidth, kernelHeight: kernelHeight, xStride: xStride, yStride: yStride, xPadding: xPadding, yPadding: yPadding)
        return OperatorNode(inputNodes: [data], name: name, operator: op)
    }
    
    public class func variable(name: String) -> VariableNode {
        return VariableNode(name: name)
    }

    public class func join(_ data: [Node], name: String? = nil) -> OperatorNode {
        let op = JoinOperator()
        return OperatorNode(inputNodes: data, name: name ?? Node.uniqueName(withPrefix: "join"), operator: op)
    }
    
    public class func reshape(_ data: Node, name: String? = nil, shape: [Int]) -> OperatorNode {
        let op = ReshapeOperator(shape: shape)
        return OperatorNode(inputNodes: [data], name: name ?? Node.uniqueName(withPrefix: "reshape"), operator: op)
    }
    
    public class func flatten(_ data: Node, name: String? = nil) -> OperatorNode {
        let op = FlattenOperator()
        return OperatorNode(inputNodes: [data], name: name ?? Node.uniqueName(withPrefix: "flatten"), operator: op)
    }
}


public class VariableNode: Node {
    
    init(name: String, attributes: [String: Any]? = nil) {
        super.init(name: name, attributes: attributes)
    }

    public required init?(coder decoder: NSCoder) {
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
    }
}


public class OperatorNode: Node {
    
    public var op: Operator
    
    init(inputNodes: [Node] = [], name: String?, operator op: Operator) {
        self.op = op
        super.init(inputNodes: inputNodes, name: name)
    }
    
    public required init?(coder decoder: NSCoder) {
        self.op = decoder.decodeObject(forKey: "op") as! Operator
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(op, forKey: "op")
    }
}

public class Operator: NSObject, NSCoding {
    
    public override init() {
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        super.init()
    }
    
    public func encode(with coder: NSCoder) {
    }
    
    public func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        fatalError()
    }
    
    public func forward(input: [NDArray?]) -> NDArray {
        fatalError()
    }
    
    public func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        fatalError()
    }
}


public class ElementWiseOperator: Operator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        precondition(!(inShapes[0] == nil && inShapes[1] == nil))
        if inShapes[0] != nil && inShapes[1] != nil {
            precondition(inShapes[0]! == inShapes[1]!)
        }
        let lhsShape = inShapes[0] == nil ? inShapes[1] : inShapes[0]
        let rhsShape = inShapes[1] == nil ? inShapes[0] : inShapes[1]
        let outShapes = (input: [lhsShape, rhsShape], output: lhsShape!)
        return outShapes
    }
}

public class AddOperator: ElementWiseOperator {
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let lhs = input[0]!
        let rhs = input[1]!
        return add(lhs, rhs)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let lhs = input[0]!
        let rhs = input[1]!
        
        let lhsGradient = NDArray(shape: lhs.shape, elements: delta.elements)
        let rhsGradient = NDArray(shape: rhs.shape, elements: delta.elements)
        return [lhsGradient, rhsGradient]
    }
}


public class SubtractOperator: ElementWiseOperator {
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let lhs = input[0]!
        let rhs = input[1]!
        return subtract(lhs, rhs)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let lhs = input[0]!
        let rhs = input[1]!
        
        let lhsGradient = NDArray(shape: lhs.shape, elements: delta.elements)
        let rhsGradient = NDArray(shape: rhs.shape, elements: delta.elements.map { -$0 })
        return [lhsGradient, rhsGradient]
    }
}


public class MultiplyOperator: ElementWiseOperator {
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let lhs = input[0]!
        let rhs = input[1]!
        return multiply(lhs, rhs)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let lhs = input[0]!
        let rhs = input[1]!
        
        let lhsGradient = multiply(rhs, delta)
        let rhsGradient = multiply(lhs, delta)
        return [lhsGradient, rhsGradient]
    }
}


public class DivideOperator: ElementWiseOperator {
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let lhs = input[0]!
        let rhs = input[1]!
        return divide(lhs, rhs)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let lhs = input[0]!
        let rhs = input[1]!
        
        let lhsGradient = multiply(NDArray(shape: rhs.shape, elements: rhs.elements.map { 1 / $0 }), delta)
        let rhsGradient = multiply(NDArray(shape: rhs.shape, elements: zip(lhs.elements, rhs.elements).map { -($0 / ($1 * $1)) }), delta)
        return [lhsGradient, rhsGradient]
    }
}


public class DotOperator: Operator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        precondition(inShapes[0] != nil && inShapes[1] != nil)
        let lhsShape = inShapes[0]
        let rhsShape = inShapes[1]
        let outputShape: Shape
        if lhsShape!.count == 2 && rhsShape!.count == 2 {
            outputShape = [lhsShape![0], rhsShape![1]]
        } else if lhsShape!.count == 2 && rhsShape!.count == 1 {
            outputShape = [lhsShape![0]]
        } else if lhsShape!.count == 1 && rhsShape!.count == 2 {
            outputShape = [rhsShape![1]]
        } else if lhsShape!.count == 1 && rhsShape!.count == 1 {
            outputShape = [1]
        } else {
            fatalError()
        }
        let outShapes = (input: [lhsShape, rhsShape], output: outputShape)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let lhs = input[0]!
        let rhs = input[1]!
        return dot(lhs, rhs)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let lhs = input[0]!
        let rhs = input[1]!
        
        if lhs.shape.count == 2 && rhs.shape.count == 1 {
            let lhsGradient = outer(delta, rhs)
            let rhsGradient = gemv(lhs, delta, transposeA: true)
            return [lhsGradient, rhsGradient]
        } else if lhs.shape.count == 1 && rhs.shape.count == 2 {
            let lhsGradient = gemv(rhs, delta)
            let rhsGradient = outer(lhs, delta)
            return [lhsGradient, rhsGradient]
        } else if lhs.shape.count == 1 && rhs.shape.count == 1 {
            let lhsGradient = multiply(rhs, delta.elements[0])
            let rhsGradient = multiply(lhs, delta.elements[0])
            return [lhsGradient, rhsGradient]
        }
        
        let lhsGradient = gemm(delta, rhs, transposeB: true)
        let rhsGradient = gemm(lhs, delta, transposeA: true)
        return [lhsGradient, rhsGradient]
    }
}


public class LinearRegressionOutputOperator: Operator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let dataShape = inShapes[0]
        let targetShape = inShapes[0]
        let outShapes = (input: [dataShape, targetShape], output: inShapes[0]!)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let input = input[0]!
        return NDArray(shape: input.shape, elements: input.elements)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let output = input[0]!
        let target = input[1]!
        return [subtract(output, target)]
    }
}


public class LogisticRegressionOutputOperator: Operator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let dataShape = inShapes[0]
        let labelShape = inShapes[0]
        let outShapes = (input: [dataShape, labelShape], output: inShapes[0]!)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let input = input[0]!
        return NDArray(shape: input.shape, elements: input.elements.map { sigmoid($0) })
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let target = input[1]!
        return [subtract(output, target)]
    }
}


public class SoftmaxOutputOperator: Operator {

    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let dataShape = inShapes[0]
        let labelShape = inShapes[0]
        let outShapes = (input: [dataShape, labelShape], output: inShapes[0]!)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let shape = input[0]!.shape
        let size = input[0]!.size
        let input = input[0]!.elements
        let maxValue = input.max()!
        let shiftedInput = input.map { $0 - maxValue }
        
        var sum: Float32 = 0.0
        for i in 0..<shiftedInput.count {
            sum += exp(shiftedInput[i])
        }
        var elements = [Float32](repeating: 0.0, count: size)
        for i in 0..<shiftedInput.count {
            elements[i] = exp(shiftedInput[i]) / sum
        }
        
        return NDArray(shape: shape, elements: elements)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let target = input[1]!
        return [subtract(output, target)]
    }
}


public class ReinforceOperator: SoftmaxOutputOperator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let dataShape = inShapes[0]
        let labelShape = inShapes[0]
        let outShapes = (input: [dataShape, labelShape], output: inShapes[0]!)
        return outShapes
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let reward = input[2]!
        let gradients = super.backward(delta: delta, input: input, output: output)
        return [outer(gradients[0]!, reward).reshaped(gradients[0]!.shape)]
    }
}


public enum ActivationFunction: Int {
    
    case rectifiedLinear
    case sigmoid
    case tanh
    
    var forwardOperatorFunction: (Float32) -> Float32 {
        switch self {
        case .sigmoid:
            return { 1.0 / (1.0 + exp(-$0)) }
        case .tanh:
            return { 1.7159 * tanhf(0.66666667 * $0) }
        case .rectifiedLinear:
            return { ($0 < 0.0) ? 0.0 : $0 }
        }
    }
    
    var backwardOperatorFunction: (Float32) -> Float32 {
        switch self {
        case .sigmoid:
            return { $0 * (1 - $0) }
        case .tanh:
            return { 0.66666667 / 1.7159 * (1.7159 + ($0)) * (1.7159 - ($0)) }
        case .rectifiedLinear:
            return { ($0 <= 0.0) ? 0.0 : 1.0 }
        }
    }
}


public class ActivationOperator: Operator {
    
    public var activationFunction: ActivationFunction
    
    init(activationFunction: ActivationFunction) {
        self.activationFunction = activationFunction
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.activationFunction = ActivationFunction(rawValue: decoder.decodeInteger(forKey: "activationFunction"))!
        super.init()
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let outShapes = (input: [inShapes[0]], output: inShapes[0]!)
        return outShapes
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(activationFunction.rawValue, forKey: "activationFunction")
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let data = input[0]!
        let operatorFunction = self.activationFunction.forwardOperatorFunction
        return NDArray(shape: data.shape, elements: data.elements.map { operatorFunction($0) })
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let inDelta = delta
        let data = input[0]!
        let operatorFunction = self.activationFunction.backwardOperatorFunction
        let outDelta = multiply(inDelta, NDArray(shape: data.shape, elements: output.elements.map { operatorFunction($0) }))
        return [outDelta]
    }
}


public class FullyConnectedOperator: Operator {
    
    public var units: Int
    
    public init(units: Int) {
        self.units = units
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.units = decoder.decodeInteger(forKey: "units")
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(units, forKey: "units")
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        precondition(inShapes[0] != nil)
        precondition(inShapes[0]!.count == 1)
        let dataShape = inShapes[0]
        let weightsShape = [units, inShapes[0]![0]]
        let biasShape = [units]
        let outShapes = (input: [dataShape, weightsShape, biasShape], output: [units])
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let data = input[0]!
        let weights = input[1]!
        let bias = input[2]!
        let output = add(gemv(weights, data), bias)
        return output
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?]  {
        let data = input[0]!
        let weights = input[1]!
        
        let weightsGradient = outer(delta, data)
        let biasGradient = delta
        let dataGradient = gemv(weights, delta, transposeA: true)

        return [dataGradient, weightsGradient, biasGradient]
    }
}


public class ConvolutionOperator: Operator {
    
    public var kernelWidth: Int
    public var kernelHeight: Int
    public var xStride: Int
    public var yStride: Int
    public var xPadding: Int
    public var yPadding: Int
    public var filters: Int
    
    public init(kernelWidth: Int, kernelHeight: Int, xStride: Int = 1, yStride: Int = 1, xPadding: Int = 0, yPadding: Int = 0, filters: Int) {
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
        self.xStride = xStride
        self.yStride = yStride
        self.xPadding = xPadding
        self.yPadding = yPadding
        self.filters = filters
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.kernelWidth = decoder.decodeInteger(forKey: "kernelWidth")
        self.kernelHeight = decoder.decodeInteger(forKey: "kernelHeight")
        self.xStride = decoder.decodeInteger(forKey: "xStride")
        self.yStride = decoder.decodeInteger(forKey: "yStride")
        self.xPadding = decoder.decodeInteger(forKey: "xPadding")
        self.yPadding = decoder.decodeInteger(forKey: "yPadding")
        self.filters = decoder.decodeInteger(forKey: "filters")
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(kernelWidth, forKey: "kernelWidth")
        coder.encode(kernelHeight, forKey: "kernelHeight")
        coder.encode(xStride, forKey: "xStride")
        coder.encode(yStride, forKey: "yStride")
        coder.encode(xPadding, forKey: "xPadding")
        coder.encode(yPadding, forKey: "yPadding")
        coder.encode(filters, forKey: "filters")
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        precondition(inShapes[0] != nil)
        precondition(inShapes[0]!.count == 3)
        let outChannels = filters
        let inChannels = inShapes[0]![0]
        let inHeight = inShapes[0]![1]
        let inWidth = inShapes[0]![2]
        let outWidth = Int((inWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        let outHeight = Int((inHeight + yPadding * 2 - kernelHeight) / yStride) + 1
        let dataShape = inShapes[0]
        let weightsShape = [outChannels, inChannels, kernelHeight, kernelWidth]
        let biasShape = [outChannels]
        let outShapes = (input: [dataShape, weightsShape, biasShape], output: [outChannels, outHeight, outWidth])
        return outShapes
    }
    
    func imageToRows(_ image: NDArray) -> NDArray {
        let channels = image.shape[0]
        let imageHeight = image.shape[1]
        let imageWidth = image.shape[2]
        let outWidth = Int((imageWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        let outHeight = Int((imageHeight + yPadding * 2 - kernelHeight) / yStride) + 1
        
        var elements = [Float32](repeating: 0, count: outWidth * outHeight * kernelWidth * kernelHeight * channels)
        
        let image = image.elements
        let channelStride = imageWidth * imageHeight
        
        var k = 0
        var m = -yPadding
        for _ in 0..<outHeight {
            var n = -xPadding
            for _ in 0..<outWidth {
                var channelOffset = 0
                for _ in 0..<channels {
                    for y in m..<(m + kernelHeight) {
                        for x in n..<(n + kernelWidth) {
                            if x >= 0 && y >= 0 && x < imageWidth && y < imageHeight {
                                elements[k] = image[channelOffset + (y * imageWidth) + x]
                            }
                            k += 1
                        }
                    }
                    channelOffset += channelStride
                }
                n += xStride
            }
            m += yStride
        }
        
        return NDArray(shape: [outWidth * outHeight, kernelWidth * kernelHeight * channels], elements: elements)
    }
    
    func rowsToImage(_ rows: NDArray, imageWidth: Int, imageHeight: Int) -> NDArray {
        let kernelWidth = self.kernelWidth
        let kernelHeight = self.kernelHeight
        
        let channels = Int(rows.shape[1] / (kernelWidth * kernelHeight))
        let outWidth = Int((imageWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        let outHeight = Int((imageHeight + yPadding * 2 - kernelHeight) / yStride) + 1
        
        var elements = [Float32](repeating: 0, count: imageWidth * imageHeight * channels)
        
        let rows = rows.elements
        let channelStride = imageWidth * imageHeight
        
        var k = 0
        var n = -yPadding
        for _ in 0..<outHeight {
            var m = -xPadding
            for _ in 0..<outWidth {
                var channelOffset = 0
                for _ in 0..<channels {
                    for y in n..<(n + kernelHeight) {
                        for x in m..<(m + kernelWidth) {
                            if x >= 0 && y >= 0 && x < imageWidth && y < imageHeight {
                                elements[channelOffset + (y * imageWidth) + x] += rows[k]
                            }
                            k += 1
                        }
                    
                    }
                    channelOffset += channelStride
                }
                m += xStride
            }
            n += yStride
        }
        return NDArray(shape: [channels, imageHeight, imageWidth], elements: elements)
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let data = input[0]!
        let weights = input[1]!
        let bias = input[2]!
        
        let outChannels = filters
        let inChannels = data.shape[0]
        let inHeight = data.shape[1]
        let inWidth = data.shape[2]
        let outWidth = Int((inWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        let outHeight = Int((inHeight + yPadding * 2 - kernelHeight) / yStride) + 1
    
        let reshapedData = imageToRows(data)
        var elements = [Float32]()
        for k in 0..<outChannels {
            elements.append(contentsOf: [Float32](repeating: bias.elements[k], count: outWidth * outHeight))
        }
        let reshapedWeights = weights.reshaped([outChannels, kernelWidth * kernelHeight * inChannels])
        let reshapedBias = NDArray(shape: [outChannels, outHeight, outWidth], elements: elements)

        let output = gemm(reshapedWeights, reshapedData, transposeB: true).reshaped([outChannels, outHeight, outWidth])
        return add(output, reshapedBias)
    }
    
    func computeBiasGradient(delta: NDArray) -> NDArray {
        let outChannels = delta.shape[0]
        let outHeight = delta.shape[1]
        let outWidth = delta.shape[2]
        
        let delta = delta.elements
        var elements = [Float32](repeating: 0.0, count: outChannels)
        var i = 0
        for k in 0..<outChannels {
            var sum: Float32 = 0.0
            for _ in 0..<outHeight {
                for _ in 0..<outWidth {
                    sum += delta[i]
                    i += 1
                }
            }
            elements[k] = sum
        }
        return NDArray(elements)
    }
    
    func computeWeightsGradient(data: NDArray, delta: NDArray) -> NDArray {
        let inChannels = data.shape[0]
        let outChannels = delta.shape[0]
        let outHeight = delta.shape[1]
        let outWidth = delta.shape[2]
        
        let reshapedData = imageToRows(data)
        let reshapedDelta = delta.reshaped([outChannels, outWidth * outHeight])
        let gradient = dot(reshapedDelta, reshapedData).reshaped([outChannels, inChannels, kernelHeight, kernelWidth])
        return gradient
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let data = input[0]!
        let weights = input[1]!

        let weightsGradient = computeWeightsGradient(data: data, delta: delta)
        let biasGradient = computeBiasGradient(delta: delta)
        
        let inChannels = data.shape[0]
        let inHeight = data.shape[1]
        let inWidth = data.shape[2]
        let outChannels = delta.shape[0]
        let outHeight = delta.shape[1]
        let outWidth = delta.shape[2]
        
        let reshapedDelta = delta.reshaped([outChannels, outWidth * outHeight])
        let reshapedWeights = weights.reshaped([outChannels, kernelWidth * kernelHeight * inChannels])
        let dataGradient = rowsToImage(gemm(reshapedDelta, reshapedWeights, transposeA: true), imageWidth: inWidth, imageHeight: inHeight)

        return [dataGradient, weightsGradient, biasGradient]
    }
}


public enum PoolingFunction: Int {
    
    case max
}

public class PoolingOperator: Operator {
    
    public var poolingFunction: PoolingFunction
    public var kernelWidth: Int
    public var kernelHeight: Int
    public var xStride: Int
    public var yStride: Int
    public var xPadding: Int
    public var yPadding: Int
    
    public init(poolingFunction: PoolingFunction,  kernelWidth: Int, kernelHeight: Int, xStride: Int = 1, yStride: Int = 1, xPadding: Int = 0, yPadding: Int = 0) {
        self.poolingFunction = poolingFunction
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
        self.xStride = xStride
        self.yStride = yStride
        self.xPadding = xPadding
        self.yPadding = yPadding
        
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.poolingFunction = PoolingFunction(rawValue: decoder.decodeInteger(forKey: "poolingFunction"))!
        self.kernelWidth = decoder.decodeInteger(forKey: "kernelWidth")
        self.kernelHeight = decoder.decodeInteger(forKey: "kernelHeight")
        self.xStride = decoder.decodeInteger(forKey: "xStride")
        self.yStride = decoder.decodeInteger(forKey: "yStride")
        self.xPadding = decoder.decodeInteger(forKey: "xPadding")
        self.yPadding = decoder.decodeInteger(forKey: "yPadding")
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(poolingFunction.rawValue, forKey: "poolingFunction")
        coder.encode(kernelWidth, forKey: "kernelWidth")
        coder.encode(kernelHeight, forKey: "kernelHeight")
        coder.encode(xStride, forKey: "xStride")
        coder.encode(yStride, forKey: "yStride")
        coder.encode(xPadding, forKey: "xPadding")
        coder.encode(yPadding, forKey: "yPadding")
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let inHeight = inShapes[0]![1]
        let inWidth = inShapes[0]![2]
        let outChannels = inShapes[0]![0]
        let outHeight = Int((inHeight + yPadding * 2 - kernelHeight) / yStride) + 1
        let outWidth = Int((inWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        let dataShape = inShapes[0]
        let outShapes = (input: [dataShape], output: [outChannels, outHeight, outWidth])
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let data = input[0]!
        
        let inChannels = data.shape[0]
        let inHeight = data.shape[1]
        let inWidth = data.shape[2]
        let outChannels = inChannels
        let outHeight = Int((inHeight + yPadding * 2 - kernelHeight) / yStride) + 1
        let outWidth = Int((inWidth + xPadding * 2 - kernelWidth) / xStride) + 1
        
        let input = data.elements

        var elements = [Float32](repeating: 0.0, count: outWidth * outHeight * outChannels)
        
        var i = 0
        for l in 0..<inChannels {
            var y = -yPadding
            for _ in 0..<outHeight {
                var x = -xPadding
                for _ in 0..<outWidth {
                    var maxValue = -FLT_MAX
                    for m in 0..<kernelHeight {
                        for n in 0..<kernelWidth {
                            if x + n >= 0 && y + m >= 0 && x + n < inWidth && y + m < inHeight {
                                let value = input[(l * inWidth * inHeight) + ((y + m) * inWidth) + (x + n)]
                                if value > maxValue {
                                    maxValue = value
                                }
                            }
                            
                        }
                    }
                    elements[i] = maxValue
                    i += 1
                    x += xStride
                }
                y += yStride
            }
        }
        
        return NDArray(shape: [outChannels, outHeight, outWidth], elements: elements)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let inDelta = delta
        let data = input[0]!
        
        let inChannels = data.shape[0]
        let inHeight = data.shape[1]
        let inWidth = data.shape[2]
        let outHeight = inDelta.shape[1]
        let outWidth = inDelta.shape[2]
        
        let input = data.elements
        let delta = inDelta.elements
        
        var elements = [Float32](repeating: 0.0, count: inWidth * inHeight * inChannels)
        
        for l in 0..<inChannels {
            var y = -yPadding
            for i in 0..<outHeight {
                var x = -xPadding
                for j in 0..<outWidth {
                    var maxValue = -FLT_MAX
                    var p: Int!
                    var q: Int!
                    for m in 0..<kernelHeight {
                        for n in 0..<kernelWidth {
                            if x + n >= 0 && y + m >= 0 && x + n < inWidth && y + m < inHeight {
                                let value = input[(l * inWidth * inHeight) + ((y + m) * inWidth) + (x + n)]
                                if value > maxValue {
                                    maxValue = value
                                    p = y + m
                                    q = x + n
                                }
                            }
                            
                        }
                    }
                    elements[(l * inWidth * inHeight) + (p * inWidth) + q] = delta[(l * outWidth * outHeight) + (i * outWidth) + j]
                    x += xStride
                }
                y += yStride
            }
        }
        
        let outDelta = NDArray(shape: [inChannels, inHeight, inWidth], elements: elements)
        return [outDelta]
    }
}


public class ReshapeOperator: Operator {
    
    public var shape: [Int]
    
    public init(shape: [Int]) {
        self.shape = shape
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        self.shape = decoder.decodeObject(forKey: "shape") as! [Int]
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
        coder.encode(shape, forKey: "shape")
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let outShapes = (input: [inShapes[0]], output: shape)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let input = input[0]!
        return input.reshaped(shape)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let inDelta = delta
        let input = input[0]!
        let outDelta = inDelta.reshaped(input.shape)
        return [outDelta]
    }
}


public class FlattenOperator: Operator {
    
    public override init() {
        super.init()
    }
    
    public required init?(coder decoder: NSCoder) {
        super.init(coder: decoder)
    }
    
    public override func encode(with coder: NSCoder) {
        super.encode(with: coder)
    }
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        let outShapes = (input: [inShapes[0]], output: [inShapes[0]!.reduce(1, *)])
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        let input = input[0]!
        return input.reshaped([input.size])
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        let inDelta = delta
        let input = input[0]!
        let outDelta = inDelta.reshaped(input.shape)
        return [outDelta]
    }
}


public class JoinOperator: Operator {
    
    public override func inferShapes(inShapes: [Shape?]) -> (input: [Shape?], output: Shape) {
        var outputShape = inShapes[0]!
        for i in 1..<inShapes.count {
            outputShape[0] += inShapes[i]![0]
        }
        let outShapes = (input: [inShapes[0]], output: outputShape)
        return outShapes
    }
    
    public override func forward(input: [NDArray?]) -> NDArray {
        var elements = [Float32]()
        var shape = input[0]!.shape
        elements.append(contentsOf: input[0]!.elements)
        for i in 1..<input.count {
            elements.append(contentsOf: input[i]!.elements)
            assert(shape.count == input[i]!.shape.count)
            shape[0] += input[i]!.shape[0]
        }
        return NDArray(shape: shape, elements: elements)
    }
    
    public override func backward(delta: NDArray, input: [NDArray?], output: NDArray) -> [NDArray?] {
        var gradients = [NDArray]()
        var start = 0
        for i in 0..<input.count {
            let size = input[i]!.size
            let shape = input[i]!.shape
            let end = start + size
            let elements = Array(delta.elements[start..<end])
            gradients.append(NDArray(shape: shape, elements: elements))
            start = end
        }
        return gradients
    }
}


public class ShapeInferencer {
    
    var graph: Graph
    
    public init(graph: Graph) {
        self.graph = graph
    }
    
    public func inferShapes(givenShapes: [String: Shape]) -> (input: [String: [Shape?]], output: [String: Shape?]) {
        var inShapes = [String: [Shape?]]()
        var inputShapes = [String: [Shape?]]()
        var outputShapes = [String: Shape?]()
        for node in graph.forwardSortedNodes {
            if node is VariableNode && givenShapes[node.name!] == nil {
                continue
            }
            let outputShape: Shape
            if node is VariableNode {
                outputShape = givenShapes[node.name!]!
                outputShapes[node.name!] = outputShape
            } else {
                let outShapes = (node as! OperatorNode).op.inferShapes(inShapes: inShapes[node.name!]!)
                outputShape = outShapes.output
                inputShapes[node.name!] = outShapes.input
                outputShapes[node.name!] = outputShape
            }
            let successors = graph.successors(of: node)
            for i in 0..<successors.count {
                let successor = successors[i]
                var shapes = inShapes[successor.name!] ?? [Shape?](repeating: nil, count: successor.inputNodes.count)
                let index = successor.inputNodes.index(of: node)!
                shapes[index] = outputShape
                inShapes[successor.name!] = shapes
            }
        }
        return (input: inputShapes, output: outputShapes)
    }
}

public class Executor {
    
    public var bindings: [String: NDArray]!
    public var graph: Graph
    
    public init(graph: Graph) {
        self.graph = graph
    }
    
    public func forward() -> (inputs: [String: [NDArray?]], outputs: [String: NDArray]) {
        var inputs = [String: [NDArray?]]()
        var outputs = [String: NDArray]()
        
        for node in graph.forwardSortedNodes {
            if node is VariableNode && bindings[node.name!] == nil { // unbound variable
                continue
            }
            let output: NDArray!
            if node is VariableNode {
                output = bindings[node.name!]
            } else {
                output = (node as! OperatorNode).op.forward(input: inputs[node.name!]!)
            }
            outputs[node.name!] = output
            let successors = graph.successors(of: node)
            for i in 0..<successors.count {
                let successor = successors[i]
                var input = inputs[successor.name!] ?? [NDArray?](repeating: nil, count: successor.inputNodes.count)
                let index = successor.inputNodes.index(of: node)!
                input[index] = output
                inputs[successor.name!] = input
            }
        }
        
        return (inputs: inputs, outputs: outputs)
    }
    
    public func backward(inputs: [String: [NDArray?]], outputs: [String: NDArray]) -> [String: NDArray] {
        var deltas = [String: NDArray]()
        var gradients = [String: NDArray]()
        
        for node in graph.backwardSortedNodes {
            if node is VariableNode {
                if deltas[node.name!] != nil {
                    let delta = deltas[node.name!]!
                    gradients[node.name!] = delta
                }
                continue
            }
            let input = inputs[node.name!]!
            let output = outputs[node.name!]!
            let delta = deltas[node.name!] ?? NDArray(shape: output.shape, elements: [Float32](repeating: 1.0, count: output.size))
            let gradients = (node as! OperatorNode).op.backward(delta: delta, input: input, output: output)
            let predecessors = graph.predecessors(of: node)
            for i in 0..<gradients.count {
                let predecessors = predecessors[i]
                var delta = deltas[predecessors.name!]
                delta = delta == nil ? gradients[i] : add(delta!, gradients[i]!)
                deltas[predecessors.name!] = delta
            }
        }
        return gradients
    }
}


public class Graph: NSObject, NSCoding {
    
    var backwardGraph: [Node: [Node]]!
    var forwardGraph: [Node: [Node]]!
    
    public var outputNodes: [Node]
    
    public lazy var forwardSortedNodes: [Node] = {
        var graph = self.backwardGraph!
        
        var queue = Array(graph.keys.filter() {
            return graph[$0]!.count == 0
        })
        
        var result = [Node]()
        while queue.count > 0 {
            let value = queue.popLast()!
            result.append(value)
            for key in self.forwardGraph[value]! {
                graph[key] = graph[key]!.filter { $0 != value }
                if graph[key]!.count == 0 {
                    queue.append(key)
                }
            }
        }
        
        return result
    }()
    
    public lazy var backwardSortedNodes: [Node] = {
        return self.forwardSortedNodes.reversed()
    }()
    
    public lazy var variableNodes: [VariableNode] = {
        return self.forwardSortedNodes.filter { $0 is VariableNode } as! [VariableNode]
    }()
    
    public lazy var operatorNodes: [OperatorNode] = {
        return self.forwardSortedNodes.filter { $0 is OperatorNode } as! [OperatorNode]
    }()
    
    public func successors(of node: Node) -> [Node] {
        return forwardGraph[node]!
    }
    
    public func predecessors(of node: Node) -> [Node] {
        return backwardGraph[node]!
    }
    
    func buildGraph(roots: [Node]) -> [Node: [Node]] {
        var map = [Node: [Node]]()
        var visitedNodes = Set([Node]())
        var queue = roots
        while queue.count > 0 {
            let current = queue.remove(at: 0)
            map[current] = current.inputNodes
            let successors = current.inputNodes
            for successor in successors {
                if !visitedNodes.contains(successor) {
                    visitedNodes.insert(successor)
                    queue.append(successor)
                }
            }
        }
        return map
    }
    
    public init(outputNodes: [Node]) {
        self.outputNodes = outputNodes
        super.init()
        backwardGraph = buildGraph(roots: outputNodes)
        forwardGraph = backwardGraph.reversed()
    }
    
    public required convenience init?(coder decoder: NSCoder) {
        let outputNodes = decoder.decodeObject(forKey: "outputNodes") as! [Node]
        self.init(outputNodes: outputNodes)
    }
    
    public func encode(with coder: NSCoder) {
        coder.encode(outputNodes, forKey: "outputNodes")
    }
}


public class Network: NSObject, NSCoding {
    
    public var graph: Graph!
    
    public var parameters = [String: NDArray]()
 
    var executor: Executor!

    public convenience init(outputNodes: [Node]) {
        let graph = Graph(outputNodes: outputNodes)
        self.init(graph: graph)
    }
    
    public init(graph: Graph) {
        self.graph = graph
        self.executor = Executor(graph: graph)
        super.init()
    }
    
    public required convenience init?(coder decoder: NSCoder) {
        let graph = decoder.decodeObject(forKey: "graph") as! Graph
        self.init(graph: graph)
        self.parameters = decoder.decodeObject(forKey: "parameters") as! [String: NDArray]
    }
    
    public func encode(with coder: NSCoder) {
        coder.encode(graph, forKey: "graph")
        coder.encode(parameters, forKey: "parameters")
    }

    func initializeParameters(inputShapes: [String: Shape]) {
        let inferredShapes = ShapeInferencer(graph: graph).inferShapes(givenShapes: inputShapes)
        
        let inputNames = Set(inputShapes.keys)

        var parameterShapes = [Shape]()
        var parameterNodes = [Node]()
        for node in graph.operatorNodes {
            for i in 0..<node.inputNodes.count {
                let inputNode = node.inputNodes[i]
                if inputNode is VariableNode && !inputNames.contains(inputNode.name!) {
                    parameterShapes.append(inferredShapes.input[node.name!]![i]!)
                    parameterNodes.append(inputNode)
                }
            }
        }
        
        for (node, shape) in zip(parameterNodes, parameterShapes) {
            let initializer: (() -> NDArray)
            switch node.attributes["initializer"] as? String {
            case .some("xavier"):
                initializer = { return xavier(shape: shape) }
            default:
                initializer = { return zero(shape: shape) }
            }
            parameters[node.name!] = initializer()
        }
    }
    
    public func fit(data: [String: [NDArray]], evaluationData: [String: [NDArray]]? = nil, batchSize: Int, learningRate: Float32, momentum: Float32 = 1.0, epochs: Int, callback: (() -> ())? = nil) {
        var data = data
        
        var inputShapes = [String: Shape]()
        for (key, value) in data {
            inputShapes[key] = value.first!.shape
        }
        initializeParameters(inputShapes: inputShapes)
        
        if data.keys.count == 0 {
            return
        }
        
        var cachedGradients = [String: NDArray]()
        
        let dataSize = data[data.keys.first!]!.count
        for epoch in 0..<epochs {
            shuffle(&data)
            
            for i in 0..<Int(dataSize / batchSize) {
                let batchStartIndex = i * batchSize
                let batchEndIndex = (i + 1) * batchSize
                
                var accumulatedGradients = [String: NDArray]()
                
                for (name, value) in parameters {
                    accumulatedGradients[name] = NDArray(shape: value.shape)
                }
                
                for i in batchStartIndex..<batchEndIndex {
                    var bindings = parameters
                    for (key, array) in data {
                        bindings[key] = array[i]
                    }
                    executor.bindings = bindings
                    
                    let (inputs, outputs) = executor.forward()
                    let gradients = executor.backward(inputs: inputs, outputs: outputs)
                    
                    for (name, _) in parameters {
                        accumulatedGradients[name] = add(accumulatedGradients[name]!, gradients[name]!)
                    }
                }
                
                for (name, value) in parameters {
                    let gradient: NDArray
                    if momentum == 1.0 {
                        gradient = multiply(accumulatedGradients[name]!, -(learningRate / Float32(batchSize)))
                    } else {
                        gradient = add(multiply(accumulatedGradients[name]!, -(learningRate / Float32(batchSize))), multiply(cachedGradients[name] ?? NDArray(shape: value.shape), momentum))
                    }
                    cachedGradients[name] = gradient
                    parameters[name] = add(value, gradient)
                }
            }
            
            if evaluationData != nil {
                let metrics = graph.outputNodes.map { outputNode -> EvaluationMetric in
                    precondition(outputNode is OperatorNode)
                    switch (outputNode as! OperatorNode).op {
                    case is SoftmaxOutputOperator:
                        return EvaluationMetric.accuracy
                    case is LogisticRegressionOutputOperator:
                        return EvaluationMetric.accuracy
                    case is LinearRegressionOutputOperator:
                        return EvaluationMetric.meanAbsoluteError
                    default:
                        fatalError()
                    }
                }
                
                let results = self.evaluate(data: evaluationData!, metrics: metrics)
                
                var metricCounts = [EvaluationMetric: Int]()
                
                let metricStrings = zip(metrics, results).map { metric, result -> String in
                    let metricName: String
                    switch metric {
                    case .accuracy:
                        metricName = "accuracy"
                    case .meanAbsoluteError:
                        metricName = "mean absolute error"
                    }
                    let count = metricCounts[metric] ?? 0
                    let string = metrics.filter { $0 == metric }.count > 1 ? "\(metricName) \(count + 1): \(result)" : "\(metricName): \(result)"
                    metricCounts[metric] = count + 1
                    return string
                }
                print("epoch: \(epoch + 1) \(metricStrings.joined())")           
            }
            
            if callback != nil {
                callback!()
            }
        }
    }

    public func evaluate(data: [String: [NDArray]], metrics: [EvaluationMetric]) -> [Float32] {
        let dataSize = data[data.keys.first!]!.count
        
        var predicted = graph.outputNodes.map { _ in [NDArray]() }
        var actual = graph.outputNodes.map { _ in [NDArray]() }
        
        for i in 0..<dataSize {
            var bindings = parameters
            for (key, array) in data {
                bindings[key] = array[i]
            }
            executor.bindings = bindings
            var (_, outputs) = executor.forward()
            for j in 0..<graph.outputNodes.count {
                let node = graph.outputNodes[j]
                predicted[j].append(outputs[node.name!]!)
                actual[j].append(outputs[node.inputNodes[1].name!]!)
            }
        }

        let metrics = zip(graph.outputNodes, metrics).map { (outputNode, metric) -> ([NDArray], [NDArray]) -> Float32 in
            switch metric {
            case .accuracy:
                return (outputNode as! OperatorNode).op is SoftmaxOutputOperator ? EvaluationMetricFunctions.Multiclass.accuracy : EvaluationMetricFunctions.Binary.accuracy
            case .meanAbsoluteError:
                return EvaluationMetricFunctions.meanAbsoluteError
            }
        }
        
        var results = [Float32]()
        for i in 0..<graph.outputNodes.count {
            results.append(metrics[i](predicted[i], actual[i]))
        }
        return results
    }
    
    public func predict(data: [String: [NDArray]]) -> [[NDArray]] {
        let dataSize = data[data.keys.first!]!.count
        
        var output = graph.outputNodes.map { _ in [NDArray]() }
        
        for i in 0..<dataSize {
            var bindings = parameters
            for (key, array) in data {
                bindings[key] = array[i]
            }
            executor.bindings = bindings
            var (_, outputs) = executor.forward()
            for j in 0..<graph.outputNodes.count {
                let node = graph.outputNodes[j]
                 output[j].append(outputs[node.name!]!)
            }
        }
        return output
    }
}
