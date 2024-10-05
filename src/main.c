#include "value.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    val *a = mgc_value(7.5);
    val *b = mgc_value(1.5);
    val *c = mgc_add(a, b);
    val *d = mgc_value(2.0);
    val *e = mgc_mul(c, d);

    /* loss definition goes here */

    struct mgc_val **sorted = calloc(50, sizeof(struct mgc_val *));
    ptrdiff_t n = mgc_toposort(sorted, e);


    double h = 0.1;
    for (ptrdiff_t i = 0; i < 5; ++i) {
        mgc_zero_gradient(sorted, n);
        e->grad = 1.0;
        mgc_forward(sorted, n);
        mgc_backward(sorted, n);
        mgc_sgd(sorted, n, h);
        mgc_print_graph(e);
    }
    free(sorted);
    return 0;
}
