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
    struct mgc_ref_vec sorted;
    mgc_ref_vec_init(&sorted);
    ptrdiff_t n = mgc_toposort(&sorted, l);

    /* constant learning rate for simplicity */
    double h = 0.05;

    /* gradient descent over a, b */
    for (ptrdiff_t i = 0; i < 25; ++i) {
        /* forward pass */
        mgc_forward(&sorted);
        /* backward pass */
        mgc_zero_gradient(&sorted);
        l->grad = 1.0;
        mgc_backward(&sorted);
        /* parameter update */
        a->value -= h * a->grad;
        b->value -= h * b->grad;
        printf("loss=%g, a=%g, x=%g, b=%g\n", l->value, a->value, x->value,
               b->value);
    }
    return 0;
}
