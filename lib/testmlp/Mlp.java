package testmlp;


import java.util.ArrayList;
import java.util.concurrent.atomic.DoubleAccumulator;

public class Mlp {
    public ArrayList<Integer> getNpl() {
        return npl;
    }

    public Integer getL() {
        return l;
    }

    public ArrayList<ArrayList<ArrayList<Double>>> getW() {
        return w;
    }

    public ArrayList<ArrayList<Double>> getDeltas() {
        return deltas;
    }

    public ArrayList<ArrayList<Double>> getX() {
        return x;
    }

    public ArrayList<Integer> npl;
    public Integer l;
    public ArrayList<ArrayList<ArrayList<Double>>> w ;
    public ArrayList<ArrayList<Double>> deltas;
    public ArrayList<ArrayList<Double>> x;

    public Mlp(ArrayList<Integer> npl) {
        this.npl = npl;
        this.l = npl.size() - 1;
        this.w = new ArrayList<>();
        this.w.add(new ArrayList<>());
        for (int layer = 1 ; layer < l+1; layer++){
            this.w.add(new ArrayList<>());
            for ( int i = 0; i < npl.get(layer - 1) +1; i ++){
                this.w.get(layer).add(new ArrayList<>());
                for(int j = 0; j <npl.get(layer) +1; j++){
                    this.w.get(layer).get(i).add(Math.random() * 2.0 - 1.0);
                }
            }
        }
        this.deltas = new ArrayList<>();
        this.deltas.add(new ArrayList<>());
        for(int layer = 1; layer < l + 1; layer++){
            this.deltas.add(new ArrayList<>());
            for(int j = 0; j < npl.get(layer) + 1; j++){
                this.deltas.get(layer).add(0.0);
            }
        }
        this.x = new ArrayList<>();
        for(int layer = 0; layer < l +1; layer++){
            this.x.add(new ArrayList<>());
            for(int j = 0; j < npl.get(layer) + 1; j++){
                if(j ==  0){
                    this.x.get(layer).add(1.0);
                }else{
                    this.x.get(layer).add(0.0);
                }
            }
        }
    }
    public Mlp mlpCreate(ArrayList<Integer> npl){
        return new Mlp(npl);
    }
    public ArrayList<Double> mlpPredictCommon(Mlp mlp, ArrayList<Double> sampleInputs, boolean classification){
        for(int j = 1; j < mlp.getNpl().get(0) +1; j++){
            Double value = sampleInputs.get(j-1);
            mlp.x.get(0).set(j,value);
        }
        for(int layer = 1; layer < mlp.l +1; layer++){
            for (int j = 1; j < mlp.getNpl().get(layer) +1; j++){
                double result = 0.0;
                for(int i = 0; i< mlp.getNpl().get(layer -1) +1; i++){
                    result += mlp.getW().get(layer).get(i).get(j) * mlp.getX().get(layer-1).get(i);
                }
                if(layer != mlp.getL() || classification){
                    result = Math.tanh(result);
                }
                mlp.getX().get(layer).set(j,result);
            }
        }
        //RETOUR QUE LISTE AVEC LE PREMIER ELEMENT
        ArrayList retour = new ArrayList();
        retour.add(mlp.getX().get(mlp.getL()).get(1));
        return retour;
    }
    public ArrayList<Double> mlpPredictClassification(Mlp mlp,ArrayList<Double> sampleInputs){
        return mlpPredictCommon(mlp,sampleInputs,true);
    }
    public ArrayList<Double> mlpPredictRegression(Mlp mlp,ArrayList<Double> sampleInputs){
        return mlpPredictCommon(mlp,sampleInputs,false);
    }
    public void mlpTrainCommun(Mlp mlp, ArrayList<ArrayList<Double>> datasetInputs, ArrayList<ArrayList<Double>> datasetExpectedOutputs, int iterationCount, Double alpha, boolean classification){
        for(int it = 0; it < iterationCount; it++){
            int k = (int)(Math.random() * datasetInputs.size());
            mlpPredictCommon(mlp,datasetInputs.get(k),classification);
            for(int j = 1; j<mlp.getNpl().get(mlp.getL())+1;j++){
                mlp.deltas.get(mlp.getL()).set(j,mlp.getX().get(mlp.getL()).get(j));
                if(classification)
                    mlp.getDeltas().get(mlp.getL()).set(j,mlp.getDeltas().get(mlp.getL()).get(j)* (1- mlp.getX().get(mlp.getL()).get(j)* mlp.getX().get(mlp.getL()).get(j-1)));
            }
            for (int layer = 2; layer > mlp.l+1; layer--){
                for(int i = 1; i < mlp.getNpl().get(layer-1) +1; i++){
                    Double result = 0.0;
                    for(int j = 1; j < mlp.npl.get(layer) +1; j++){
                        result += mlp.getW().get(layer).get(i).get(j) * mlp.getDeltas().get(layer).get(j);
                    }
                    result *= 1 - mlp.x.get(layer -1).get(i) * mlp.x.get(layer -1).get(i);
                    mlp.getDeltas().get(layer-1).set(i,result);
                }
            }
            for(int layer = 1; layer < mlp.getL() + 1; layer++){
                for(int i = 0; i<mlp.getNpl().get(layer - 1) + 1; i++){
                    for (int j = 1; j < mlp.getNpl().get(layer) + 1; j++){
                        Double bis = mlp.getW().get(layer).get(i).get(j);
                        bis -= alpha * mlp.getX().get(layer-1).get(i) * mlp.getDeltas().get(layer).get(j);
                        mlp.getW().get(layer).get(i).set(j,bis);
                    }
                }
            }
        }
    }
    public void mlpTrainClassification(Mlp mlp, ArrayList<ArrayList<Double>> datasetInputs, ArrayList<ArrayList<Double>> datasetExpectedOutputs, int iterationCount, Double alpha){
        mlpTrainCommun(mlp,datasetInputs,datasetExpectedOutputs,iterationCount,alpha,true);
    }
    public void mlpTrainRegression(Mlp mlp, ArrayList<ArrayList<Double>> datasetInputs, ArrayList<ArrayList<Double>> datasetExpectedOutputs, int iterationCount, Double alpha){
        mlpTrainCommun(mlp,datasetInputs,datasetExpectedOutputs,iterationCount,alpha,false);
    }
}
