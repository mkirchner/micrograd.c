#include <stdio.h>
#include "value.h"


int main(int argc, char* argv[])
{
    val *a = mgc_value(7.5);
    printf("%g\n", a->value);
    return 0;
}
