import Foundation

public struct MNIST {
    
    public static func loadImages(path: String) -> [[UInt8]] {
        let initialOffset = 16
        let imageStride = 784
        
        var result = [[UInt8]]()
  
        let downloadsDirectory = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
        let url = downloadsDirectory.appendingPathComponent(path)
        
        let file = try! FileHandle(forReadingFrom: url)
        
        var offset: UInt64 = UInt64(initialOffset)
        while true {
            file.seek(toFileOffset: offset)
            let data: Data = file.readData(ofLength: imageStride)
            if data.count == 0 {
                break
            }
            var bytes = [UInt8](repeating: 0, count: imageStride)
            data.copyBytes(to: &bytes, count: bytes.count)
            result.append(bytes)
            offset += UInt64(imageStride)
        }
        
        file.closeFile()
        
        return result
    }
    
    public static func loadLabels(path: String) -> [UInt8] {
        let initialOffset = 8
        
        var result = [UInt8]()
        
        let downloadsDirectory = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
        let url = downloadsDirectory.appendingPathComponent(path)
        
        let file = try! FileHandle(forReadingFrom: url)
        
        var offset: UInt64 = UInt64(initialOffset)
        while true {
            file.seek(toFileOffset: offset)
            let data: Data = file.readData(ofLength: 1)
            if data.count == 0 {
                break
            }
            var bytes = [UInt8](repeating: 0, count: 1)
            data.copyBytes(to: &bytes, count: bytes.count)
            result.append(bytes.first!)
            offset += UInt64(1)
        }
        
        file.closeFile()
        
        return result
    }
    
    public static func loadData(imagesPath: String, labelsPath: String) -> [String: [NDArray]] {
        let input = loadImages(path: imagesPath).map { (image) -> NDArray in
            return NDArray(shape: [784], elements: image.map { Float($0) / 255.0 })
        }
        let labels = loadLabels(path: labelsPath).map { (label) -> NDArray in
            var elements = [Float](repeating: 0.0, count: 10)
            elements[Int(label)] = 1.0
            return NDArray(shape: [10], elements: elements)
        }
        return ["input": input, "labels": labels]
    }
    
    public static func loadTrainingData() -> [String: [NDArray]] {
        return loadData(imagesPath: "train-images.idx3-ubyte", labelsPath: "train-labels.idx1-ubyte")
    }
    
    public static func loadTestData() -> [String: [NDArray]] {
        return loadData(imagesPath: "t10k-images.idx3-ubyte", labelsPath: "t10k-labels.idx1-ubyte")
    }
    
    /*
     Please download and unzip these files before executing run:
     http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
     http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
     http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
     http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    */
    
    public static func run() {
        let input = Node.variable(name: "input")
        let labels = Node.variable(name: "labels")
        var output = Node.fullyConnected(input, units: 128)
        output = Node.activation(output, activationFunction: .rectifiedLinear)
        output = Node.fullyConnected(output, units: 10)
        output = Node.softmaxOutput(output, labels)
        let network = Network(outputNodes: [output])
        
        let trainingData = loadTrainingData()
        let testData = loadTestData()
        
        network.fit(data: trainingData, evaluationData: testData, batchSize: 10, learningRate: 0.1, epochs: 3)
        
        print(" ")
       
        let evaluationData = ["input": Array(testData["input"]![0..<10]), "labels": Array(testData["labels"]![0..<10])]
        let results = network.predict(data: evaluationData).first!
        
        for i in 0..<10 {
            let image = evaluationData["input"]![i]
            let predicted = results[i]
            let actual = evaluationData["labels"]![i]
            let predictedLabel = predicted.elements.index(of: predicted.elements.max()!)!
            let actualLabel = actual.elements.index(of: actual.elements.max()!)!
            
            for j in stride(from: 0, to: 784, by: 28) {
                print(image.elements[j..<j + 28].map { $0 > 0.0 ? "##" : ".." }.joined())
            }
            print("predicted label:")
            print(predictedLabel)
            print("actual label:")
            print(actualLabel)
            print(" ")
        }
    }
}
