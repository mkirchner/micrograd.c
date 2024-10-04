#include "value.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    val *a = mgc_value(7.5);
    val *b = mgc_value(1.5);
    val *c = mgc_add(a, b);
    val *d = mgc_value(2.0);
    val *e = mgc_mul(c, d);
    // printf("%g\n", e->value);

    e->grad = 1.0;
    mgc_backward(e);
    mgc_print_graph(e);
    return 0;
}
