package testmlp;

import java.util.ArrayList;

public class test {

    public static void main(String[] args) {
        ArrayList liste = new ArrayList() {{
            add(2);
            add(2);
            add(1);
        }};
        Mlp mlp = new Mlp(liste);

        ArrayList<ArrayList<Double>> xTrain = new ArrayList<>();
        ArrayList Table = new ArrayList();
        Table.add(0.0);
        Table.add(0.0);
        ArrayList Table1 = new ArrayList();
        Table1.add(1.0);
        Table1.add(0.0);
        ArrayList Tabl2 = new ArrayList();
        Tabl2.add(0.0);
        Tabl2.add(1.0);
        ArrayList Tabl3 = new ArrayList();
        Tabl3.add(1.0);
        Tabl3.add(1.0);

        xTrain.add(Table);
        xTrain.add(Table1);
        xTrain.add(Tabl2);
        xTrain.add(Tabl3);



        ArrayList<ArrayList<Double>> yTrain = new ArrayList<>();
        ArrayList Ta = new ArrayList();
        Ta.add(-1.0);
        ArrayList Ta1 = new ArrayList();
        Ta1.add(1.0);
        ArrayList Ta2 = new ArrayList();
        Ta2.add(1.0);

        ArrayList Ta3 = new ArrayList();
        Ta3.add(-1.0);

        yTrain.add(Ta);
        yTrain.add(Ta1);
        yTrain.add(Ta2);
        yTrain.add(Ta2);

        mlp.mlpTrainClassification(mlp, xTrain, yTrain, 100000, 0.1);
        System.out.println(mlp.mlpPredictClassification(mlp, xTrain.get(0)));
        System.out.println(mlp.mlpPredictClassification(mlp, xTrain.get(1)));
        System.out.println(mlp.mlpPredictClassification(mlp, xTrain.get(2)));
        System.out.println(mlp.mlpPredictClassification(mlp, xTrain.get(3)));

    }
}
