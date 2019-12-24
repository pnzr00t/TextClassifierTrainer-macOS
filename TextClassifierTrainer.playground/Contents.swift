// macOS playgorund
import Foundation
import CreateML



func readFileAndSplitByFile(baseFileDataUrl: URL, trainingDataFileURL: URL, testingDataFileURL: URL) -> Bool {
    do { // clear files with data tranin and test data
        try "".write(to: trainingDataFileURL, atomically: true, encoding: .utf8)
        try "".write(to: testingDataFileURL, atomically: true, encoding: .utf8)
    } catch {
        fatalError("Cant clear train and test file \(error.localizedDescription)" )
    }
    
    guard let trainingFileHandle = try? FileHandle(forWritingTo: trainingDataFileURL),
    let testingFileHandle = try? FileHandle(forWritingTo: testingDataFileURL) else {
        fatalError("Error! Could not load resource files.")
    }
    
    testingFileHandle.write(("[").data(using: .utf8)!)
    trainingFileHandle.write(("[").data(using: .utf8)!)

    
    if freopen(baseFileDataUrl.path, "r", stdin) == nil {
        perror(baseFileDataUrl.path)
        return false
    }
    
    var testingLineCount: Int = 0
    var trainingLineCount: Int = 0
    
    while let line = readLine() {
        print(line)
        
        if Int.random(in: 0..<100) > 70 {
            testingFileHandle.seekToEndOfFile()
            if testingLineCount > 0 {
                testingFileHandle.write(("," + "\n" + line).data(using: .utf8)!)
            } else {
                testingFileHandle.write(("\n" + line).data(using: .utf8)!)
            }
                
            print("Testing data: ", line)
            
            testingLineCount += 1
        } else {
            trainingFileHandle.seekToEndOfFile()
            if trainingLineCount > 0 {
                trainingFileHandle.write(("," + "\n" + line).data(using: .utf8)!)
            } else {
                trainingFileHandle.write(("\n" + line).data(using: .utf8)!)
            }
            print("Training data: ", line)
            
            trainingLineCount += 1
        }
    }
    
    testingFileHandle.write(("\n]").data(using: .utf8)!)
    trainingFileHandle.write(("\n]").data(using: .utf8)!)
    trainingFileHandle.closeFile()
    testingFileHandle.closeFile()
    
    print("trainingDataFileURL = ", trainingDataFileURL)
    print("testingDataFileURL = ", testingDataFileURL)
    return true
}

// get URL to the the documents directory in the sandbox
let documentsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

let trainingDataFileURL = documentsUrl.appendingPathComponent("training-set.json")
let testingDataFileURL = documentsUrl.appendingPathComponent("testing.json")

guard let amazonInstatnVideoUrl = Bundle.main.url(forResource: "reviews_Amazon_Instant_Video_5", withExtension: "json") else {
    fatalError("Cant reda Amazon input file")
}
    
    // Split file by training and Data set
    readFileAndSplitByFile(baseFileDataUrl: amazonInstatnVideoUrl, trainingDataFileURL: trainingDataFileURL, testingDataFileURL: testingDataFileURL)



/*guard let trainingDataFileURL = Bundle.main.url(forResource: "amazon-reviews", withExtension: "json"),
let testingDataFileURL = Bundle.main.url(forResource: "testing-reviews", withExtension: "json") else {
    fatalError("Error! Could not load resource files.")
}*/

var trainingDataTable: MLDataTable? = nil
var testingDataTable: MLDataTable? = nil

do {
    trainingDataTable = try MLDataTable(contentsOf: trainingDataFileURL)
    testingDataTable = try MLDataTable(contentsOf: testingDataFileURL)
    
    print("Entries used for training: \(trainingDataTable?.size)")
    print("Entries used for testing: \(testingDataTable?.size)")
} catch {
    print("Upload training data exception - ", error.localizedDescription)
    exit(0)
}

print("\n", "==========================")


do {
    guard let trainingDataTable = trainingDataTable, let testingDataTable = testingDataTable else { print("Cant instantiate training or testig data table"); exit(0); }
    
    
    
    let mlTextClassifier = try MLTextClassifier(trainingData: trainingDataTable, textColumn: "reviewText", labelColumn: "overall")
    
    let trainingAccuracy = (1.0 - mlTextClassifier.trainingMetrics.classificationError) * 100
    let validateAccuracy = (1.0 - mlTextClassifier.validationMetrics.classificationError) * 100
    
    
    let evaluateMetrics = mlTextClassifier.evaluation(on: testingDataTable, textColumn: "reviewText", labelColumn: "overall")
    let evaluateAccuracy = (1.0 - evaluateMetrics.classificationError) * 100
    
    print("Training accuracy - ", trainingAccuracy)
    print("Validate accuracy - ", validateAccuracy)
    print("Evaludate accuracy - ", evaluateAccuracy)
    
    print("\n")
    print("mlTextClassifier.trainingMetrics.classificationError = ",mlTextClassifier.trainingMetrics.classificationError)
    print("mlTextClassifier.validationMetrics.classificationError = ", mlTextClassifier.validationMetrics.classificationError)
    print("evaluateMetrics.classificationError = ", evaluateMetrics.classificationError)

    
    let metaDataForModel = MLModelMetadata(author: "POA",
                                           shortDescription: "A model trained to classify product review sentiment", version: "0.2")

    
    // get URL to the the documents directory in the sandbox
    let documentsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    // add a filename
    let outModelFileUrl = documentsUrl.appendingPathComponent("ReviewClassifier.mlmodel")
    
    //let modelFileURL = URL(fileURLWithPath: "/Volumes/Samsung_T5/Users/TestFolder/CoreML/NauralLanguage/TextClassifierTrainer/TrainedTextClassifierModel/ReviewClassifier.mlmodel")
    try mlTextClassifier.write(to: outModelFileUrl, metadata: metaDataForModel)
    print("Model file path = ", outModelFileUrl);
} catch {
    print("Training info exception - ", error.localizedDescription)
    exit(0)
}
print("This is the end")


