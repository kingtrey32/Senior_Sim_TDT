package com.example.final_project;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    private Bitmap bitmap = null;
    private Module module = null;

    String [] items = {"T-shirt/top", "Trouser","Pullover", "Dress",
            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};


    //loadButton
    Button loadButton;
    //classifyButton
    Button classifyButton;
    //textView
    TextView textView;
    //imageView
    ImageView imageView;



    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //loadButton
        loadButton = findViewById(R.id.loadButton);
        //classifyButton
        classifyButton = findViewById(R.id.classifyButton);
        //textView
        textView = findViewById(R.id.textView);
        //imageView
        imageView = findViewById(R.id.imageView);


        try {
            module = Module.load(assetFilePath(this, "model_scripted_3_Channel.pt"));
        } catch (IOException e) {
            Log.e("PTMobileWalkthru", "Error reading assets", e);
            finish();
        }

        //get user image
        //    loads into imageView
        loadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent();
                i.setType("image/*");
                i.setAction(Intent.ACTION_GET_CONTENT);

                launchSomeActivity.launch(i);
            }
        });




        //classify button goes here

        classifyButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                float[] model_STD = {1.0f, 1.0f, 1.0f};
                float[] model_MEAN = {0.0f, 0.0f, 0.0f};

                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, model_MEAN,  model_STD);
                //final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                // running the model
                //error says that somewhere in this chained call, a string is present when it should be a tuple
                final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

                // getting tensor content as java array of floats
                final float[] scores = outputTensor.getDataAsFloatArray();

                // searching for the index with maximum score
                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;
                for (int i = 0; i < scores.length; i++) {
                    if (scores[i] > maxScore) {
                        maxScore = scores[i];
                        maxScoreIdx = i;
                    }
                }


                String className = items[maxScoreIdx];

                // showing className on UI
                textView.setText(className);
            }
        });
    }

    //loadButton activity result, gets image
    ActivityResultLauncher<Intent> launchSomeActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null) {
                        Uri selectedImageUri = data.getData();
                        try {
                            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                            imageView.setImageBitmap(bitmap);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });


}