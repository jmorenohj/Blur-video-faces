
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace cv;
using namespace std;
#define SHARED_SIZE 125

//*(arr+3*i*1280+3*j+2)
__global__ void blur(uchar *frame,int x, int y, int height, int width, int kernelSize, int totalRows, int totalCols){
    

    __shared__ uchar temp[SHARED_SIZE*SHARED_SIZE*3];
    int initial_row = y-kernelSize/2;
    for(int col = x-kernelSize/2;col<=x+width+kernelSize/2;col++){
        temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3] = frame[(initial_row+threadIdx.x)*3*totalCols+col*3];
        temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3+1] = frame[(initial_row+threadIdx.x)*3*totalCols+col*3+1];
        temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3+2] = frame[(initial_row+threadIdx.x)*3*totalCols+col*3+2];
    }
    __syncthreads();
    uchar acumR = 0, acumG = 0, acumB = 0;
    for(int col = 0;col<=width+kernelSize-1;col++){
        if(col>=kernelSize){
            temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize/2)*3] = acumR;
            temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize/2)*3+1] = acumG;
            temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize/2)*3+2] = acumB;
            acumR -= temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize)*3]/kernelSize;
            acumG -= temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize)*3+1]/kernelSize;
            acumB -= temp[threadIdx.x*SHARED_SIZE*3 + (col-kernelSize)*3+2]/kernelSize;
        }
        acumR += temp[threadIdx.x*SHARED_SIZE*3+col*3]/kernelSize;
        acumG += temp[threadIdx.x*SHARED_SIZE*3+col*3+1]/kernelSize;
        acumB += temp[threadIdx.x*SHARED_SIZE*3+col*3+2]/kernelSize;
    }
    __syncthreads();
    acumR = 0, acumG = 0, acumB = 0;
    for(int row = 0;row<=height+kernelSize-1;row++){
        if(row>=kernelSize){
            temp[(row-kernelSize/2)*SHARED_SIZE*3 + threadIdx.x*3] = acumR;
            temp[(row-kernelSize/2)*SHARED_SIZE*3 + threadIdx.x*3+1] = acumG;
            temp[(row-kernelSize/2)*SHARED_SIZE*3 + threadIdx.x*3+2] = acumB;
            acumR -= temp[(row-kernelSize)*SHARED_SIZE*3 + threadIdx.x*3]/kernelSize;
            acumG -= temp[(row-kernelSize)*SHARED_SIZE*3 + threadIdx.x*3+1]/kernelSize;
            acumB -= temp[(row-kernelSize)*SHARED_SIZE*3 + threadIdx.x*3+2]/kernelSize;
        }
        acumR += temp[row*SHARED_SIZE*3+threadIdx.x*3]/kernelSize;
        acumG += temp[row*SHARED_SIZE*3+threadIdx.x*3+1]/kernelSize;
        acumB += temp[row*SHARED_SIZE*3+threadIdx.x*3+2]/kernelSize;
    }
    __syncthreads();
    
    
    for(int col = x-kernelSize/2;col<=x+width+kernelSize/2;col++){
        frame[(initial_row+threadIdx.x)*3*totalCols+col*3] = temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3] ;
        frame[(initial_row+threadIdx.x)*3*totalCols+col*3+1] = temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3+1] ;
        frame[(initial_row+threadIdx.x)*3*totalCols+col*3+2] = temp[threadIdx.x*SHARED_SIZE*3+(col-(x-kernelSize/2))*3+2] ;
    }
}

int main(int argc, char *argv[]){
    // Se definen los directorios en donde se lee y se escribe
    char path[100] = "";
    char path2[100] = "";
    strcat(path,argv[1]);
    strcat(path2,argv[2]);

    // Almacena parámetros del video de entrada
    VideoCapture cap(path);
    int frame_width = (int)(cap.get(3));
    int frame_height = (int)(cap.get(4));
    Size frame_size(frame_width, frame_height);
    int fps = 20;
    int totalFrames = (int)cap.get(7);

    // Establece parámetros del video de salida
    VideoWriter output(path2, VideoWriter::fourcc('M', 'P', '4', 'V'),fps, frame_size);

    cout<<"Total frames "<<totalFrames<<endl;
    cout<<"Frame width "<<frame_width<<endl;
    cout<<"Frame height "<<frame_height<<endl;

    // Verifica si se abrió el video con éxito
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    

    // Verifica si se abrió el video con éxito
    int kernelSize = 15;
    double div = (double)1/(kernelSize*kernelSize);


    
    // Variable para almacenar coordenadas y medidas de rostros
    vector<Rect> faces;
    CascadeClassifier face_cascade;
    face_cascade.load("/content/build/blur/haarcascade_frontalface_alt.xml");
    cudaError_t err = cudaSuccess;
    // Se itera por todos los frames del video
    int cont = 0;
    int blocks_num = 1;

    while(cap.isOpened()){
        // Establece fotograma a analizar
        Mat frame;
        bool isSuccess = cap.read(frame);
        
        // Verifica si el frame se leyó con éxito
        if (!isSuccess){
            cout << "Stream disconnected" << endl;
            break;
        }
            
        if (frame.empty())break;
        // Detecta los rostros en el fotograma
        face_cascade.detectMultiScale(frame, faces, 1.1, 3,0);
        // Itera sobre todos los rostros detectados
        if(faces.empty()){
          output.write(frame);
          continue;
        }

        uchar *h_frame, *d_frame;
        int size,threads_num = 0;
        h_frame = frame.isContinuous()? frame.data: frame.clone().data;
        uint length = frame.total()*frame.channels();        
        size = sizeof(uchar)*length;
        err = cudaMalloc((void **)&d_frame, size);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(d_frame, h_frame, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to copy vector h_frame from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        //cout<<faces.size()<<endl;
        for(Rect r:faces){
            if(r.height+kernelSize>=SHARED_SIZE || r.width+kernelSize>=SHARED_SIZE)continue;
            //cout<<r.height<<" "<<r.width<<endl;
            threads_num = r.height+kernelSize-1;
            blur<<<blocks_num,threads_num>>>(d_frame,r.x,r.y,r.height,r.width,kernelSize,frame_height,frame_width);
        }
        cudaDeviceSynchronize();
       
        err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(h_frame, d_frame, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        

        err = cudaFree(d_frame);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        
    
        
        
        // Limpia el vector de rostros
        faces.clear();
        output.write(frame);

        // Muestra porcentaje de avance
        //cout<<(double)cont*100/totalFrames<<"%"<<endl;
        cont++;
        

        
    }
    
    err = cudaDeviceReset();
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Finaliza el programa
    
    cap.release();
    destroyAllWindows();
    return 0;
}
