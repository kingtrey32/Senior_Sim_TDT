package com.example.myapplication;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    int SELECT_PICTURE=200;

    ImageView imageView;
    TextView textView;
    Button loadbutton;
    Button classifybutton;

    String [] items = {"T-shirt/top", "Trouser","Pullover", "Dress",
            "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

    Module module;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            module = Module.load(assetFilePath(MainActivity.this));
        } catch (IOException e) {
            e.printStackTrace();
        }

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
        loadbutton = findViewById(R.id.loadbutton);
        classifybutton = findViewById(R.id.classifybutton);

        loadbutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                imageChooser();
            }
        });

        classifybutton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                classifyImage(((BitmapDrawable) imageView.getDrawable()).getBitmap());
            }
        });
    }

    private void imageChooser() {
        Intent i = new Intent();
        i.setType("image/*");
        i.setAction(Intent.ACTION_GET_CONTENT);

        launchSomeActivity.launch(i);
    }

    ActivityResultLauncher<Intent> launchSomeActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    Intent data = result.getData();
                    if (data != null) {
                        Uri selectedImageUri = data.getData();
                        Bitmap selectedImageBitmap;
                        try {
                            selectedImageBitmap = MediaStore.Images.Media.getBitmap(
                                    this.getContentResolver(),
                                    selectedImageUri);
                            imageView.setImageBitmap(selectedImageBitmap);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });

    private void classifyImage(Bitmap bitmap) {
        // Preprocess the image
        Tensor inputTensor = preprocessImage(bitmap);

        // Pass the input tensor to the model to get the output
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // Get the predicted class index
        float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
            }
        }

        // Return the predicted class label
    }

    private Tensor preprocessImage(Bitmap bitmap) {
        // Resize the image to 28x28 pixels
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);

        // Convert the image to a float array
        int[] pixels = new int[28 * 28];
        resizedBitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);
        float[] floatValues = new float[28 * 28];
        for (int i = 0; i < pixels.length; i++) {
            final int val = pixels[i];
            floatValues[i] = (float) (((val >> 16) & 0xFF) * 0.299 + ((val >> 8) & 0xFF) * 0.587 + (val & 0xFF) * 0.114);
        }



        // Normalize the pixel values
            float[] mean = {0.5f};
            float[] std = {0.5f};
            for (int i = 0; i < floatValues.length; i++) {
                floatValues[i] = (floatValues[i] / 255.0f - mean[0]) / std[0];
            }

            // Create a PyTorch Tensor from the float array
            long[] shape = {1, 1, 28, 28};
            return Tensor.fromBlob(floatValues, shape);
        }
    private String assetFilePath(Context context) throws IOException {
        File file = new File(context.getFilesDir(), "fashion_mnist_cnn_traced.pt");
        if (!file.exists()) {
            try (InputStream is = context.getAssets().open("fashion_mnist_cnn_traced.pt")) {
                try (FileOutputStream os = new FileOutputStream(file)) {
                    byte[] buffer = new byte[4 * 1024];
                    int read;
                    while ((read = ((InputStream) is).read(buffer)) != -1) {
                        os.write(buffer, 0, read);
                    }
                    os.flush();
                }
            }
        }
        return file.getPath();
    }
    }


