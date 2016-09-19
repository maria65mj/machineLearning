package ml.classifiers;

import java.util.ArrayList;
import java.util.Set;
import java.text.DecimalFormat;

import ml.DataSet;
import ml.Example;

public class WeightedPerceptronClassifier implements Classifier {

	private ArrayList<Double> w = new ArrayList<Double>();
	private ArrayList<Double> u = new ArrayList<Double>();
	private int b;
	private int b2;
	private int updated;
	private int total;
	private DataSet data;
	private int numIterations = 10;

	public WeightedPerceptronClassifier(Set<Integer> allFeatureIndices) {
		// TODO Auto-generated constructor stub
	}

	public WeightedPerceptronClassifier() {
	}

	/**
	 * @param data
	 */
	@Override
	public void train(DataSet data) {
		this.data = data;
		int size = data.getAllFeatureIndices().size();
		zeroFill(w, size);
		zeroFill(u, size);
		b = 0;
		b2 = 0;
		updated = 0;
		total = 0;

		ArrayList<Example> examples = data.getData();
		for (int i = 0; i < numIterations; i++) {
			total = 0;
			for (Example ex : examples) {
				double label = ex.getLabel();
				int prediction = b + getSum(ex);
				if (prediction * label <= 0) { // if we misclassify ex
					// update our final, weighted weights
					for (int j = 0; j < u.size(); j++)
						u.set(j, u.get(j) + updated * w.get(j));

					b2 += updated * b;

					// update all the perceptron weights
					for (int wi = 0; wi < w.size(); wi++) {
						double newVal = label * ex.getFeature(wi);
						w.set(wi, (w.get(wi) + newVal));
					}

					b += label;

					updated = 0;
				} // end of misclassify
				updated++;
				total++;
			}

			// TODO do one last weighted update here of the u and b2 weights
			// based on the final weights

		}
		// divide all of the aggregate weights by the total num of examples
		for (int n = 0; n < u.size(); n++)
			u.set(n, u.get(n) / total);

		b2 = b2 / total;
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
			sum += example.getFeature(m) * w.get(m);
		return sum;
	}

	private void zeroFill(ArrayList<Double> array, int size) {
		for (int i = 0; i < size; i++) {
			array.add(0.0);
		}
	}

	@Override
	public double classify(Example example) {
		double pred = 0;
        for (int i = 0; i < example.getFeatureSet().size(); i++) {
            pred += example.getFeature(i) * u.get(i);
        }
        System.out.println(pred + b2);

        return pred + b2;
	}

	public void setIterations(int numIterations) {
		this.numIterations = numIterations;

	}

	public String toString() {
		String result = "";

		for (int i = 0; i < data.getAllFeatureIndices().size(); i++) {
			result += i + ":" + w.get(i) + " ";
		}

		return result + b2;
	}

	public static void main(String[] args) {
		 DataSet data = new DataSet
				 ("/Users/mjm72013/Desktop/_Fall16/MachineLearning/Assignments/Assignment3/Perceptrons/simple1.csv");
		// DataSet data = new
		// DataSet("/home/mmartinez/Documents/CS158/Perceptrons/src/titanic-train.perc.csv");
//		DataSet data = new DataSet(
//				"/Users/mjm72013/Desktop/_Fall16/MachineLearning/Assignments/Assignment3/Perceptrons/titanic-train.perc.csv");
		WeightedPerceptronClassifier temp = new WeightedPerceptronClassifier(
				data.getAllFeatureIndices());
		temp.train(data);
		System.out.println(temp.toString());

		// System.out.println("Prediction: " +
		// temp.classify(data.getData().get(5)));
		//
		// // Splitting the data
		DataSet[] split = data.split(0.8);
		WeightedPerceptronClassifier titanic = new WeightedPerceptronClassifier();
		titanic.train(split[0]);

		ArrayList<Double> preds = new ArrayList<Double>();

		for (int i = 0; i < 100; i++) {
			Double correct = 0.0;
			for (Example example : split[1].getData()) {
				if (example.getLabel() == titanic.classify(example))
					correct++;
				correct = correct / split[1].getData().size();
			}
//			preds.add(Math.floor(correct * 100));
//			preds.add(new DecimalFormat("##.##").format(correct));
			preds.add(correct);
			
		}
		System.out.println(preds);
	}

}
