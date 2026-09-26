#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <Eigen/Dense>
// ---------------------------------------------------------
// ALGORITMO DEIM (Reducción de Orden)
// ---------------------------------------------------------
std::vector< int > compute_deim_indices(const Eigen::MatrixXd& U) {
int N = U.rows();
int m = U.cols();
std::vector< int > p(m);
int max_idx;
U.col(0).cwiseAbs().maxCoeff(&max_idx);
p[0] = max_idx;

for (int i = 1; i < m; ++i) {
    Eigen::MatrixXd PtU(i, i);
    for (int j = 0; j < i; ++j) {
        PtU.row(j) = U.row(p[j]).head(i);
    }
    Eigen::VectorXd Ptu(i);
    for (int j = 0; j < i; ++j) {
        Ptu(j) = U(p[j], i);
    }
    Eigen::VectorXd c = PtU.partialPivLu().solve(Ptu);
    Eigen::VectorXd r = U.col(i) - U.leftCols(i) * c;
    r.cwiseAbs().maxCoeff(&max_idx);
    p[i] = max_idx;
}
return p;
}
// ---------------------------------------------------------
// PARTE 1: Ecuación de Advección Lineal
// ---------------------------------------------------------
const double L_1 = 10.0;
const int N_1 = 200;
const double c_1 = 1.0;
const double dx_1 = L_1 / N_1;

double init_square(double x) {
if (x >= 4.0 && x <= 6.0) return 1.0;
return 0.0;
}

double exact_square(double x, double t) {
double shift = std::fmod(x - c_1 * t, L_1);
if (shift < 0) shift += L_1;
return init_square(shift);
}

void solve_advection(double nu, double t_end, const std::string& filename) {
double dt = nu * dx_1 / c_1;
int num_steps = std::ceil(t_end / dt);

std::vector< double > x(N_1);
std::vector< double > u_ftcs(N_1), u_ftcs_new(N_1);
std::vector< double > u_lax(N_1), u_lax_new(N_1);
std::vector< double > u_upwind(N_1), u_upwind_new(N_1);

for (int i = 0; i < N_1; ++i) {
    x[i] = i * dx_1;
    double val = init_square(x[i]);
    u_ftcs[i] = u_lax[i] = u_upwind[i] = val;
}

for (int step = 0; step < num_steps; ++step) {
    for (int i = 0; i < N_1; ++i) {
        int izq = (i - 1 + N_1) % N_1;
        int der = (i + 1) % N_1;

        u_ftcs_new[i] = u_ftcs[i] - 0.5 * nu * (u_ftcs[der] - u_ftcs[izq]);
        u_lax_new[i] = 0.5 * (u_lax[der] + u_lax[izq]) - 0.5 * nu * (u_lax[der] - u_lax[izq]);
        u_upwind_new[i] = u_upwind[i] - nu * (u_upwind[i] - u_upwind[izq]);
    }
    std::swap(u_ftcs, u_ftcs_new);
    std::swap(u_lax, u_lax_new);
    std::swap(u_upwind, u_upwind_new);
}

std::ofstream file(filename);
file << "x,exact,ftcs,lax,upwind\n";
for (int i = 0; i < N_1; ++i) {
    file << x[i] << "," << exact_square(x[i], t_end) << "," 
         << u_ftcs[i] << "," << u_lax[i] << "," << u_upwind[i] << "\n";
}
file.close();
std::cout << "Guardado: " << filename << std::endl;

}

// ---------------------------------------------------------
// PARTE 2: PRUEBA COMPARATIVA DEIM VS FOM
// ---------------------------------------------------------
void test_deim_string(double A, double t_end, int m_modos, const std::string& filename) {
const double L_2 = 10.0;
const double c_2 = 1.0;
const double sigma = 0.5;
const int N_2 = 200;
const int N_in = N_2 - 2; // Nodos internos
const double dx_2 = L_2 / (N_2 - 1);
const double nu = 0.3;
const double dt = nu * dx_2 / c_2;
int num_steps = std::ceil(t_end / dt);

std::vector< double > x(N_2);
std::vector< double > yL_old(N_2), yL_curr(N_2), yL_new(N_2);
std::vector< double > yFOM_old(N_2), yFOM_curr(N_2), yFOM_new(N_2);
std::vector< double > yDEIM_old(N_2), yDEIM_curr(N_2), yDEIM_new(N_2);

for (int i = 0; i < N_2; ++i) {
    x[i] = i * dx_2;
    double val = A * std::exp(-std::pow(x[i] - L_2/2.0, 2) / (2.0 * sigma * sigma));
    if (i == 0 || i == N_2 - 1) val = 0.0; 
    
    yL_old[i] = yL_curr[i] = val;
    yFOM_old[i] = yFOM_curr[i] = val;
    yDEIM_old[i] = yDEIM_curr[i] = val;
}

// --- FASE 1: Recolección de Snapshots (FOM) ---
Eigen::MatrixXd F_snap(N_in, num_steps);

for (int step = 0; step < num_steps; ++step) {
    for (int i = 1; i < N_2 - 1; ++i) {
        double dy_dx = (yFOM_curr[i+1] - yFOM_curr[i-1]) / (2.0 * dx_2);
        double factor = 1.0 / std::sqrt(1.0 + dy_dx * dy_dx); 
        F_snap(i-1, step) = factor;

        if (step == 0) {
            yFOM_new[i] = yFOM_curr[i] + 0.5 * factor * nu * nu * (yFOM_curr[i+1] - 2.0*yFOM_curr[i] + yFOM_curr[i-1]);
            yL_new[i] = yL_curr[i] + 0.5 * nu * nu * (yL_curr[i+1] - 2.0*yL_curr[i] + yL_curr[i-1]);
        } else {
            yFOM_new[i] = 2.0 * yFOM_curr[i] - yFOM_old[i] + factor * nu * nu * (yFOM_curr[i+1] - 2.0*yFOM_curr[i] + yFOM_curr[i-1]);
            yL_new[i] = 2.0 * yL_curr[i] - yL_old[i] + nu * nu * (yL_curr[i+1] - 2.0*yL_curr[i] + yL_curr[i-1]);
        }
    }
    yFOM_new[0] = yFOM_new[N_2-1] = 0.0;
    yL_new[0] = yL_new[N_2-1] = 0.0;
    
    std::swap(yFOM_old, yFOM_curr); std::swap(yFOM_curr, yFOM_new);
    std::swap(yL_old, yL_curr); std::swap(yL_curr, yL_new);
}

// --- FASE 2: SVD y Construcción de Interpolador DEIM ---
Eigen::JacobiSVD< Eigen::MatrixXd > svd(F_snap, Eigen::ComputeThinU);
Eigen::MatrixXd U = svd.matrixU().leftCols(m_modos);
std::vector< int > p_indices = compute_deim_indices(U);

Eigen::MatrixXd PtU(m_modos, m_modos);
for (int i = 0; i < m_modos; ++i) {
    PtU.row(i) = U.row(p_indices[i]);
}
Eigen::MatrixXd P_DEIM = U * PtU.inverse();

std::cout << "DEIM selecciono " << m_modos << " modos. Índices espaciales mágicos: ";
for (int idx : p_indices) std::cout << idx + 1 << " ";
std::cout << "\n";

// --- FASE 3: Simulación Online usando DEIM ---
for (int step = 0; step < num_steps; ++step) {
    // 1. Evaluar término no lineal SOLO en los m puntos DEIM
    Eigen::VectorXd f_m(m_modos);
    for (int k = 0; k < m_modos; ++k) {
        int idx = p_indices[k] + 1; // +1 porque p_indices es 0-indexed respecto a N_in
        double dy_dx = (yDEIM_curr[idx+1] - yDEIM_curr[idx-1]) / (2.0 * dx_2);
        f_m(k) = 1.0 / std::sqrt(1.0 + dy_dx * dy_dx);
    }

    // 2. Interpolar el cálculo al resto del dominio (Nx1)
    Eigen::VectorXd f_approx = P_DEIM * f_m;

    // 3. Avanzar el paso temporal
    for (int i = 1; i < N_2 - 1; ++i) {
        if (step == 0) {
            yDEIM_new[i] = yDEIM_curr[i] + 0.5 * f_approx(i-1) * nu * nu * (yDEIM_curr[i+1] - 2.0*yDEIM_curr[i] + yDEIM_curr[i-1]);
        } else {
            yDEIM_new[i] = 2.0 * yDEIM_curr[i] - yDEIM_old[i] + f_approx(i-1) * nu * nu * (yDEIM_curr[i+1] - 2.0*yDEIM_curr[i] + yDEIM_curr[i-1]);
        }
    }
    yDEIM_new[0] = yDEIM_new[N_2-1] = 0.0;
    
    std::swap(yDEIM_old, yDEIM_curr); std::swap(yDEIM_curr, yDEIM_new);
}

// --- Exportar Resultados para Comparación ---
std::ofstream file(filename);
file << "x,y_lineal,y_fom,y_deim\n";
for (int i = 0; i < N_2; ++i) {
    file << x[i] << "," << yL_curr[i] << "," << yFOM_curr[i] << "," << yDEIM_curr[i] << "\n";
}
file.close();
std::cout << "Guardado: " << filename << std::endl;

}

int main() {
std::cout << "--- Tarea 2: Solvers Iniciados ---\n";

// 1. Ecuación de Advección (Variando Courant)
solve_advection(0.5, 15.0, "adveccion_nu_0.5.csv");
solve_advection(0.8, 15.0, "adveccion_nu_0.8.csv");
solve_advection(1.0, 15.0, "adveccion_nu_1.0.csv");

// 2. Ecuación de la Cuerda (Comparación Lineal vs FOM vs DEIM)
// Utilizamos m=10 modos que evaluarán la no linealidad en solo 10 nodos espaciales en vez de 198
test_deim_string(0.1, 4.0, 10, "cuerda_comparacion_A_0.1.csv");
test_deim_string(0.5, 4.0, 10, "cuerda_comparacion_A_0.5.csv");
test_deim_string(1.0, 4.0, 10, "cuerda_comparacion_A_1.0.csv");

return 0;

}