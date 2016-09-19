package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import ml.DataSet;
import ml.Example;

public class PerceptronClassifier implements Classifier {

    private int numIterations = 10;
    private Set<Integer> featureIndices;
    private HashMap<Integer, Double> weights;
    private int b = 0;

    public PerceptronClassifier(Set<Integer> allFeatureIndices) {
        featureIndices = allFeatureIndices;
    }

    public PerceptronClassifier() {
    }

    /**
     * Trains our perceptrons
     *
     * @param data
     */
    @Override
    public void train(DataSet data) {
        Set<Integer> features = data.getAllFeatureIndices();
        int numFeatures = data.getAllFeatureIndices().size();

        // Setting all the weights to 0
        weights = new HashMap<Integer, Double>();
        for (int w = 0; w < numFeatures; w++)
            weights.put(w, 0.0);

        ArrayList<Example> examples = data.getData();

        b = 0;
        for (int i = 0; i < numIterations; i++) {
            for (Example example : examples) {
                int prediction = b + getSum(example);

                if (prediction * example.getLabel() <= 0) {// they don't agree
                    updateWeights(example);
                    b += example.getLabel();
                }
            }
        }
    }

    /**
     * Add the product of all the weights and features
     *
     * @param example
     * @return
     */
    private int getSum(Example example) {
        int sum = 0;
        for (int m = 0; m < example.getFeatureSet().size(); m++)
            sum += example.getFeature(m) * weights.get(m);
        return sum;
    }

    /**
     * Updates all the weights
     *
     * @param example
     */
    private void updateWeights(Example example) {
        for (int w = 0; w < weights.size(); w++) {
            double newVal = example.getLabel() * example.getFeature(w);
            weights.replace(w, (weights.get(w) + newVal));
        }

    }

    /**
     * Classifies the example it takes in using our trained model
     *
     * @param example
     * @return
     */
    @Override
    public double classify(Example example) {
        double total = 0;
        for (int i = 0; i < example.getFeatureSet().size(); i++) {
            total += example.getFeature(i) * weights.get(i);
        }

        return total + b;
    }

    /**
     * Sets the number of iterations
     *
     * @param numIterations
     */
    public void setIterations(int numIterations) {
        this.numIterations = numIterations;
    }

    /**
     * @return
     */
    public String toString() {
        String result = "";

        for (int i = 0; i < featureIndices.size(); i++) {
            result += i + ":" + weights.get(i) + " ";
        }

        return result + b;

    }

    public static void main(String[] args) {

        // DataSet("/home/mmartinez/Documents/CS158/Perceptrons/src/simple2.csv");
        DataSet data = new DataSet(
                "/home/mmartinez/Documents/CS158/Perceptrons/src/titanic-train.perc.csv");
        PerceptronClassifier temp = new PerceptronClassifier(
                data.getAllFeatureIndices());
        temp.train(data);
        // System.out.println(temp.toString());

        System.out.println("Prediction: " + temp.classify(data.getData().get(5)));

        // Splitting the data
        DataSet[] split = data.split(0.8);
        PerceptronClassifier titanic = new PerceptronClassifier(split[0].getAllFeatureIndices());
        titanic.train(split[0]);

        ArrayList<Integer> preds = new ArrayList<Integer>();

        for (int i = 0; i < 100; i++) {
            int correct = 0;
            for (Example example : split[1].getData()) {
                if (example.getLabel() * titanic.classify(example) > 0)
                    correct++;
                correct = correct / split[1].getData().size();
            }
            preds.add(correct);
        }

    }
}
