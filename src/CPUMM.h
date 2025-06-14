#pragma once

void CPUMM(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;

            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[col + i * N];
            }
            
            C[row * N + col] = sum;
        }
    }
}