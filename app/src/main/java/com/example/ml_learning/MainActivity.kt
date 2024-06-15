package com.example.ml_learning

import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ml_learning.ui.theme.Ml_learningTheme

class MainActivity : ComponentActivity(), ImageClassifierHelper.ClassifierListener {

    private lateinit var imageClassifierHelper: ImageClassifierHelper
    private var imageUri = mutableStateOf<Uri?>(null)
    private var isProgressVisible = mutableStateOf(false)

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            imageUri.value = it
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        imageClassifierHelper = ImageClassifierHelper(
            context = this,
            listener = this,
            colorModelFileName = "warna",
            typeModelFileName = "jenis"
        )

        setContent {
            Ml_learningTheme {
                MainActivityScreen(
                    imageUri = imageUri.value,
                    onAnalyzeClick = { analyzeImage() },
                    onGalleryClick = { selectImageFromGallery() },
                )
            }
        }
    }

    private fun analyzeImage() {
        val uri = imageUri.value
        if (uri != null) {
            isProgressVisible.value = true
            Log.d(TAG, "Starting image analysis")
            imageClassifierHelper.classifyImage(uri)
        } else {
            Log.e(TAG, "No image selected")
            Toast.makeText(this, R.string.no_image_selected, Toast.LENGTH_SHORT).show()
        }
    }

    private fun selectImageFromGallery() {
        Log.d(TAG, "Selecting image from gallery")
        galleryLauncher.launch("image/*")
    }

    override fun onFailure(error: String) {
        isProgressVisible.value = false
        Log.e(TAG, "Image classification failed: $error")
        Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
    }

    override fun onSuccess(results: List<ImageClassifierHelper.ClassificationResult>?) {
        isProgressVisible.value = false
        results?.let {
            // Handle the results (e.g., show them in a dialog or a new screen)
            val resultText = it.joinToString { result -> "${result.label}: ${result.score * 100}%" }
            Log.d(TAG, "Image classification succeeded: $resultText")
            Toast.makeText(this, resultText, Toast.LENGTH_LONG).show()
        }
    }

    override fun onModelReady() {
        Log.d(TAG, "Model is ready")
        Toast.makeText(this, R.string.model_ready, Toast.LENGTH_SHORT).show()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}

@Composable
fun MainActivityScreen(
    imageUri: Uri? = null,
    onAnalyzeClick: () -> Unit = {},
    onGalleryClick: () -> Unit = {}
) {
    val context = LocalContext.current
    var isProgressVisible by remember { mutableStateOf(false) }
    val bitmap = remember(imageUri) {
        if (imageUri != null) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(context.contentResolver, imageUri)
                ImageDecoder.decodeBitmap(source).asImageBitmap()
            } else {
                context.contentResolver.openInputStream(imageUri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream).asImageBitmap()
                }
            }
        } else {
            null
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(colorResource(id = R.color.white)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .fillMaxSize()
                .padding(32.dp)
        ) {
            if (isProgressVisible) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                )
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(top = 40.dp, bottom = 8.dp)
                    .background(
                        color = colorResource(id = R.color.white),
                        shape = RoundedCornerShape(16.dp)
                    )
            ) {
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap,
                        contentDescription = stringResource(id = R.string.app_name),
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(40.dp)
                    )
                } else {
                    Image(
                        painter = painterResource(id = R.drawable.ic_place_holder),
                        contentDescription = stringResource(id = R.string.app_name),
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(40.dp)
                    )
                }
            }

            Button(
                onClick = onAnalyzeClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 20.dp),
                shape = RoundedCornerShape(50),
                enabled = !isProgressVisible
            ) {
                Text(
                    text = stringResource(id = R.string.analyze),
                    fontSize = 20.sp
                )
            }

            Button(
                onClick = onGalleryClick,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 20.dp, bottom = 40.dp),
                shape = RoundedCornerShape(50)
            ) {
                Text(
                    text = stringResource(id = R.string.gallery),
                    fontSize = 20.sp,
                    color = Color.White
                )
            }
        }
    }
}

@Preview
@Composable
fun MainActivityPreview(){
    MainActivityScreen()
}
