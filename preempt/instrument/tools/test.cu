__global__ void test(int *i)
{
    i[0] = i[1] + 1;
    i[1] = i[2] + 1;
    i[2] = i[3] + 1;
    i[3] = i[4] + 1;
    i[4] = i[5] + 1;
    i[5] = i[6] + 1;
    i[6] = i[7] + 1;
    i[7] = i[8] + 1;
    i[8] = i[9] + 1;
    i[9] = i[10] + 1;
    i[10] = i[11] + 1;
    i[11] = i[12] + 1;
    i[12] = i[13] + 1;
    i[13] = i[14] + 1;
    i[14] = i[15] + 1;
    i[15] = i[16] + 1;
    i[16] = i[17] + 1;
    i[17] = i[18] + 1;
    i[18] = i[19] + 1;
    i[19] = i[20] + 1;
    asm("exit;");
}
