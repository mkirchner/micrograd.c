#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
    srand48(time(NULL));

    /* y_hat = a*x + b */
    val *a = mgc_value(drand48());
    val *x = mgc_value(2.0);
    val *b = mgc_value(drand48());
    val *y_hat = mgc_add(mgc_mul(a, x), b);

    /* expected y */
    val *y = mgc_value(10.0);

    /* loss function: (y - y_hat)^2 */
    val *l = mgc_pow(mgc_add(y_hat, mgc_neg(y)), mgc_value(2));

    /* topological order over compute nodes */
    struct mgc_val **sorted = calloc(50, sizeof(struct mgc_val *));
    ptrdiff_t n = mgc_toposort(sorted, l);

    /* constant learning rate for simplicity */
    double h = 0.05;

    /* gradient descent over a, b */
    for (ptrdiff_t i = 0; i < 25; ++i) {
        /* forward pass */
        mgc_forward(sorted, n);
        /* backward pass */
        mgc_zero_gradient(sorted, n);
        l->grad = 1.0;
        mgc_backward(sorted, n);
        /* parameter update */
        a->value -= h * a->grad;
        b->value -= h * b->grad;
        printf("loss=%g, a=%g, x=%g, b=%g\n", l->value, a->value, x->value,
               b->value);
    }
    free(sorted);
    return 0;
}
