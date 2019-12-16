// macOS playgorund
import Foundation
import CreateML

guard let trainingDataFileURL = Bundle.main.url(forResource: "amazon-reviews", withExtension: "json"),
let testingDataFileURL = Bundle.main.url(forResource: "testing-reviews", withExtension: "json") else {
    fatalError("Error! Could not load resource files.")
}

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
    
    
    
    let mlTextClassifier = try MLTextClassifier(trainingData: trainingDataTable, textColumn: "text", labelColumn: "label")
    
    let trainingAccuracy = (1.0 - mlTextClassifier.trainingMetrics.classificationError) * 100
    let validateAccuracy = (1.0 - mlTextClassifier.validationMetrics.classificationError) * 100
    
    
    let evaluateMetrics = mlTextClassifier.evaluation(on: testingDataTable, textColumn: "text", labelColumn: "label")
    let evaluateAccuracy = (1.0 - evaluateMetrics.classificationError) * 100
    
    print("Training accuracy - ", trainingAccuracy)
    print("Validate accuracy - ", validateAccuracy)
    print("Evaludate accuracy - ", evaluateAccuracy)
    
    print("\n")
    print("mlTextClassifier.trainingMetrics.classificationError = ",mlTextClassifier.trainingMetrics.classificationError)
    print("mlTextClassifier.validationMetrics.classificationError = ", mlTextClassifier.validationMetrics.classificationError)
    print("evaluateMetrics.classificationError = ", evaluateMetrics.classificationError)

    
    let metaDataForModel = MLModelMetadata(author: "POA",
                                           shortDescription: "A model trained to classify product review sentiment", version: "0.1")

    
    let modelFileURL = URL(fileURLWithPath: "/Volumes/Samsung_T5/Users/TestFolder/CoreML/NauralLanguage/TextClassifierTrainer/TrainedTextClassifierModel/ReviewClassifier.mlmodel")
    try mlTextClassifier.write(to: modelFileURL, metadata: metaDataForModel)

} catch {
    print("Training info exception - ", error.localizedDescription)
    exit(0)
}
print("This is the end")


