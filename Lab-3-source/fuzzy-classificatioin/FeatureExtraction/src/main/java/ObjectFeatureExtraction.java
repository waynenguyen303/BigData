import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;

import java.io.*;

/**
 * Created by Naga on 20-09-2016.
 */
public class ObjectFeatureExtraction {
    public static void main(String args[]) throws IOException {

        File[] fashionTestFolder = new File("data/fashion-test/").listFiles();
        String outputFolder = "output/";
        String[] IMAGE_CATEGORIES = {"adidas", "coat", "dress", "hat", "pants", "shirt", "shoe", "suit", "watch"};
        int x=0;
        int y=0;
        int z=0;

        for (int k=0;k<fashionTestFolder.length;k++) {

            File[] fashionTestFolderjpgs = new File("data/fashion-test/"+fashionTestFolder[k].getName()+"/").listFiles();
            String inputFolder = "data/fashion-test/"+fashionTestFolder[k].getName()+"/";

            for (int t = 0; t < fashionTestFolderjpgs.length; t++) {
                String inputImage = fashionTestFolderjpgs[t].getName();
                System.out.println(fashionTestFolderjpgs[t].getName());
                int input_class = y;
                MBFImage mbfImage = ImageUtilities.readMBF(new File(inputFolder + inputImage));
                DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();
                LocalFeatureList<Keypoint> features2 = doGSIFTEngine.findFeatures(mbfImage.flatten());
                FileWriter fw = new FileWriter(outputFolder + IMAGE_CATEGORIES[input_class] +t+ ".txt");
                BufferedWriter bw2 = new BufferedWriter(fw);
                for (int i = 0; i < features2.size(); i++) {
                    double c[] = features2.get(i).getFeatureVector().asDoubleVector();
                    bw2.write(input_class + ",");
                    for (int j = 0; j < c.length; j++) {
                        bw2.write(c[j] + " ");
                    }
                    bw2.newLine();
                }
                bw2.close();
                z++;
            }
            y++;
        }


//        FileWriter fw1 = new FileWriter(outputFolder  + "features-test.txt");
//        BufferedWriter bw1 = new BufferedWriter(fw1);


//        //to make test fashion features to one .txt --"features-test.txt"
//        for (int k=0;k<fashionTestFolder.length;k++) {
//
//            File[] fashionTestFolderjpgs = new File("data/fashion-test/"+fashionTestFolder[k].getName()+"/").listFiles();
//
//            for (int t = 0; t < fashionTestFolderjpgs.length; t++) {
//
//                String inputFolder = "data/fashion-test/"+fashionTestFolder[k].getName()+"/";
//                System.out.println(fashionTestFolderjpgs[t].getName());
//                String inputImage = fashionTestFolderjpgs[t].getName();
//
//                int input_class = y;
//                MBFImage mbfImage = ImageUtilities.readMBF(new File(inputFolder + inputImage));
//                DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();
//                LocalFeatureList<Keypoint> features1 = doGSIFTEngine.findFeatures(mbfImage.flatten());
//                for (int i = 0; i < features1.size(); i++) {
//                    double c[] = features1.get(i).getFeatureVector().asDoubleVector();
//                    bw1.write(input_class + ",");
//                    for (int j = 0; j < c.length; j++) {
//                        bw1.write(c[j] + " ");
//                    }
//                    bw1.newLine();
//                }
//
//            }
//            y++;
//        }
//        bw1.close();

        // to make the training fashion features to one .txt --- "features-train.txt"
        // **************** will rewrite code into functions so there is no duplicates later *********************

        File[] jpgs = new File("data/fashion-features/").listFiles();
        FileWriter fw = new FileWriter(outputFolder  + "features-train.txt");
        BufferedWriter bw = new BufferedWriter(fw);

        for (int k=0;k<jpgs.length;k++) {

            String inputFolder = "data/fashion-features/";
            System.out.println(jpgs[k].getName());
            String inputImage = jpgs[k].getName();

            int input_class = x;
            MBFImage mbfImage = ImageUtilities.readMBF(new File(inputFolder + inputImage));
            DoGSIFTEngine doGSIFTEngine = new DoGSIFTEngine();
            LocalFeatureList<Keypoint> features = doGSIFTEngine.findFeatures(mbfImage.flatten());

            for (int i = 0; i < features.size(); i++) {
                double c[] = features.get(i).getFeatureVector().asDoubleVector();
                bw.write(input_class + ",");
                for (int j = 0; j < c.length; j++) {
                    bw.write(c[j] + " ");
                }
                bw.newLine();
            }
            x++;
        }
        bw.close();
    }

}
