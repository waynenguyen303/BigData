from pandas_ml import ConfusionMatrix
from pandas_confusion import BinaryConfusionMatrix

if __name__ == "__main__":

    # manual entry of true and predicted values for confusion matrix demonstration
    y_true = ['man', 'man', 'woman', 'man', 'woman', 'woman', 'woman', 'man', 'man', 'woman']
    y_pred = ['woman', 'man', 'woman', 'man', 'man', 'woman', 'woman', 'man', 'woman', 'woman']

    # make a binary confusion matrix and print out stats
    con = BinaryConfusionMatrix(y_true, y_pred)
    n = y_true.__len__()
    con.print_stats()
    print("\nConfusion matrix: \n%s" % con + "\n")

    # manual calculations of some confusion matrix stats
    acc = con.ACC
    misRate = (con.FP + con.FN)/n
    truePos = (con.TP/(con.TP + con.FN))
    falsePos = (con.FP/(con.FP + con.TN))
    spec = (con.TN / (con.FP + con.TN))
    prec = (con.TP / (con.TP + con.FP))
    prev = ((con.TP + con.FN)/n)

    # print out stats
    print("Accuracy = " + str(acc))
    print("Misclassification Rate = " + str(misRate))
    print("True Positive Rate = " + str(truePos))
    print("False Positive Rate = " + str(falsePos))
    print("Specificity = " + str(spec))
    print ("Precision = " + str(prec))
    print("Prevelance = " + str(prev))



