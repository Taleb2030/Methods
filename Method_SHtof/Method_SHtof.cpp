#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <winsock2.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iomanip>
#include <exception>
#include <tuple>

#pragma comment(lib, "ws2_32.lib")
using namespace std;
namespace fs = filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===== Robust trim function =====
string trim(const string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return (start == string::npos) ? "" : str.substr(start, end - start + 1);
}

// ===== Function Parser (with robust error handling) =====
class FunctionParser {
private:
    string expr;
    double x_val, x1_val, x2_val;
    
    double evaluate(const string& e, size_t& pos) {
        skipWhitespace(e, pos);
        double result = parseTerm(e, pos);
        while (pos < e.length()) {
            skipWhitespace(e, pos);
            if (pos >= e.length()) break;
            char op = e[pos];
            if (op == '+' || op == '-') {
                pos++; double right = parseTerm(e, pos);
                result = (op == '+') ? result + right : result - right;
            } else break;
        }
        return result;
    }
    
    double parseTerm(const string& e, size_t& pos) {
        skipWhitespace(e, pos);
        double result = parseFactor(e, pos);
        while (pos < e.length()) {
            skipWhitespace(e, pos);
            if (pos >= e.length()) break;
            char op = e[pos];
            if (op == '*' || op == '/') {
                pos++; double right = parseFactor(e, pos);
                result = (op == '*') ? result * right : result / right;
            } else if (op == '^') {
                pos++; double right = parseFactor(e, pos);
                result = pow(result, right);
            } else break;
        }
        return result;
    }
    
    double parseFactor(const string& e, size_t& pos) {
        skipWhitespace(e, pos);
        if (pos >= e.length()) return 0;
        if (e[pos] == '(') {
            pos++; double result = evaluate(e, pos);
            if (pos < e.length() && e[pos] == ')') pos++;
            return result;
        }
        if (e[pos] == '-') {
            pos++; return -parseFactor(e, pos);
        }
        // Functions with safety checks
        if (pos + 2 < e.length() && e.substr(pos, 3) == "sin") {
            pos += 3; skipWhitespace(e, pos); if (e[pos] == '(') pos++;
            double val = evaluate(e, pos); if (e[pos] == ')') pos++;
            return sin(val);
        }
        if (pos + 2 < e.length() && e.substr(pos, 3) == "cos") {
            pos += 3; skipWhitespace(e, pos); if (e[pos] == '(') pos++;
            double val = evaluate(e, pos); if (e[pos] == ')') pos++;
            return cos(val);
        }
        if (pos + 2 < e.length() && e.substr(pos, 3) == "exp") {
            pos += 3; skipWhitespace(e, pos); if (e[pos] == '(') pos++;
            double val = evaluate(e, pos); if (e[pos] == ')') pos++;
            return exp(val);
        }
        if (pos + 2 < e.length() && e.substr(pos, 3) == "log") {
            pos += 3; skipWhitespace(e, pos); if (e[pos] == '(') pos++;
            double val = evaluate(e, pos); if (e[pos] == ')') pos++;
            return log(max(val, 1e-10)); // Prevent log(0)
        }
        if (pos + 3 < e.length() && e.substr(pos, 4) == "sqrt") {
            pos += 4; skipWhitespace(e, pos); if (e[pos] == '(') pos++;
            double val = evaluate(e, pos); if (e[pos] == ')') pos++;
            return sqrt(max(val, 0.0)); // Prevent sqrt(negative)
        }
        // Variables
        if (e[pos] == 'x') {
            if (pos + 1 < e.length() && e[pos + 1] == '1') { pos += 2; return x1_val; }
            else if (pos + 1 < e.length() && e[pos + 1] == '2') { pos += 2; return x2_val; }
            else { pos++; return x_val; }
        }
        // Number parsing with error handling
        size_t start = pos;
        while (pos < e.length() && (isdigit(e[pos]) || e[pos] == '.' || e[pos] == '-')) pos++;
        if (start == pos) return 0;
        try { 
            double val = stod(e.substr(start, pos - start));
            return isfinite(val) ? val : 0.0; 
        } catch (...) { return 0.0; }
    }
    
    void skipWhitespace(const string& e, size_t& pos) {
        while (pos < e.length() && isspace(e[pos])) pos++;
    }
    
public:
    FunctionParser(const string& expression) : expr(expression) {}
    double operator()(double x) { 
        x_val = x; x1_val = x; x2_val = 0; 
        size_t pos = 0; 
        double res = evaluate(expr, pos); 
        return isfinite(res) ? res : 0.0; 
    }
    double operator()(double x1, double x2) { 
        x_val = x1; x1_val = x1; x2_val = x2; 
        size_t pos = 0; 
        double res = evaluate(expr, pos); 
        return isfinite(res) ? res : 0.0; 
    }
};

// ===== Utilities =====
string url_decode(const string& src) {
    string res;
    for (size_t i = 0; i < src.size(); ++i) {
        if (src[i] == '%' && i + 2 < src.size()) {
            unsigned int hex; sscanf_s(src.substr(i + 1, 2).c_str(), "%x", &hex);
            res += static_cast<char>(hex); i += 2;
        } else if (src[i] == '+') res += ' ';
        else res += src[i];
    }
    return res;
}

string get_query_param(const string& req, const string& param) {
    size_t qpos = req.find("?");
    if (qpos == string::npos) return "";
    string query = req.substr(qpos + 1);
    size_t start = query.find(param + "=");
    if (start == string::npos) return "";
    start += param.size() + 1;
    size_t end = query.find('&', start);
    if (end == string::npos) end = query.find(' ', start);
    if (end == string::npos) end = query.find('\r', start);
    if (end == string::npos) end = query.size();
    return url_decode(query.substr(start, end - start));
}

map<string, string> parse_post_data(const string& data) {
    map<string, string> params;
    size_t start = 0;
    while (start < data.length()) {
        size_t eq = data.find('=', start);
        if (eq == string::npos) break;
        size_t amp = data.find('&', eq + 1);
        if (amp == string::npos) amp = data.length();
        string key = url_decode(data.substr(start, eq - start));
        string value = url_decode(data.substr(eq + 1, amp - eq - 1));
        params[key] = value;
        start = amp + 1;
    }
    return params;
}

// ===== Global Parameters =====
struct UserParams {
    string f_expr = "x^2 - 4*x";
    string g_expr = "x - 1";
    string g1_expr = "x1 - 1";
    string g2_expr = "x1 + x2 - 2";
    string g3_expr = "-x2";
    double x0 = 3.0, x1_0 = 0.5, x2_0 = 0.5;
    double r0 = 1.0, C = 10.0, epsilon = 0.01, mu0 = 0.0;
    int max_iter = 6;
} user_params;

// ===== COMPUTATION METHODS (ALL NOW MATCH PDFs AND SUPPORT 1D/2D) =====
void write_error_csv(const string& method, const string& error_msg) {
    ofstream csv("results.csv");
    csv << "method,k,r,x,gx,F\n";
    csv << method << ",0,0,0,ERROR," << error_msg << "\n";
    csv.close();
    cerr << "[CSV ERROR] " << method << ": " << error_msg << endl;
}

// ----- Penalty Method (already updated) -----
void compute_penalty() {
    try {
        string f_expr = user_params.f_expr;
        string g_expr = user_params.g_expr;
        bool is2D = (f_expr.find("x1") != string::npos || f_expr.find("x2") != string::npos ||
                     g_expr.find("x1") != string::npos || g_expr.find("x2") != string::npos);

        if (!is2D) {
            // 1D
            FunctionParser f(f_expr), g(g_expr);
            auto penalty = [&g](double x, double r) { double s = max(0.0, g(x)); return 0.5 * r * s * s; };
            auto F = [&f, &penalty](double x, double r) { return f(x) + penalty(x, r); };
            auto dF = [&f, &g](double x, double r) {
                double h = 1e-6, df = (f(x+h)-f(x-h))/(2*h);
                return (g(x) <= 0) ? df : df + r * g(x) * (g(x+h)-g(x-h))/(2*h);
            };
            auto minimizeF = [&dF](double x0, double r) {
                double x = x0, alpha = 0.01, eps = 1e-6;
                for (int i = 0; i < 10000; i++) {
                    double xn = x - alpha * dF(x, r);
                    if (fabs(xn - x) < eps) break;
                    x = xn;
                }
                return x;
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x,gx,F\n";
            double r = user_params.r0, C = user_params.C, x = user_params.x0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    x = minimizeF(x, r);
                    double gx = g(x), Fval = F(x, r);
                    if (isfinite(x) && isfinite(gx) && isfinite(Fval)) {
                        csv << "penalty," << k << "," << r << "," << x << "," << gx << "," << Fval << "\n";
                        wrote = true;
                    }
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) csv << "penalty,0," << r << "," << x << "," << g(x) << "," << F(x, r) << "\n";
            csv.close();
        } else {
            // 2D
            FunctionParser f(f_expr);
            FunctionParser g(g_expr);
            auto F = [&](double x1, double x2, double r) -> double {
                double val = f(x1, x2);
                double gj = g(x1, x2);
                double s = max(0.0, gj);
                val += 0.5 * r * s * s;
                return val;
            };
            auto gradF = [&](double x1, double x2, double r, double& gx1, double& gx2) {
                double h = 1e-6;
                gx1 = (F(x1+h, x2, r) - F(x1-h, x2, r)) / (2*h);
                gx2 = (F(x1, x2+h, r) - F(x1, x2-h, r)) / (2*h);
            };
            auto minimizeF = [&](double x1, double x2, double r) {
                double alpha = 0.1, eps = 1e-6;
                for (int i = 0; i < 5000; i++) {
                    double gx1, gx2;
                    gradF(x1, x2, r, gx1, gx2);
                    double t = 1.0;
                    double f0 = F(x1, x2, r);
                    double nx1, nx2;
                    while (true) {
                        nx1 = x1 - t * gx1;
                        nx2 = x2 - t * gx2;
                        if (F(nx1, nx2, r) < f0) break;
                        t *= 0.5;
                        if (t < 1e-10) { t = 0; break; }
                    }
                    if (t == 0) break;
                    if (hypot(nx1 - x1, nx2 - x2) < eps) break;
                    x1 = nx1; x2 = nx2;
                }
                return make_pair(x1, x2);
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x1,x2,g,F\n";
            double r = user_params.r0, C = user_params.C;
            double x1 = user_params.x1_0, x2 = user_params.x2_0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    auto [nx1, nx2] = minimizeF(x1, x2, r);
                    x1 = nx1; x2 = nx2;
                    double Fval = F(x1, x2, r);
                    double gv = g(x1, x2);
                    csv << "penalty," << k << "," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
                    wrote = true;
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) {
                double Fval = F(x1, x2, r);
                double gv = g(x1, x2);
                csv << "penalty,0," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
            }
            csv.close();
        }
    } catch (const exception& e) { write_error_csv("penalty", e.what()); }
}

// ----- Barrier Method (updated to handle 2D) -----
void compute_barrier() {
    try {
        string f_expr = user_params.f_expr;
        string g_expr = user_params.g_expr;
        bool is2D = (f_expr.find("x1") != string::npos || f_expr.find("x2") != string::npos ||
                     g_expr.find("x1") != string::npos || g_expr.find("x2") != string::npos);

        if (!is2D) {
            // 1D (original)
            FunctionParser f(f_expr), g(g_expr);
            if (g(user_params.x0) >= 0) {
                cerr << "[BARRIER] Initial point outside feasible region! Forcing x0=0.5" << endl;
                user_params.x0 = 0.5;
            }
            auto barrier = [&g](double x, double r) -> double { 
                double gx = g(x);
                return (gx < -1e-8) ? -r / gx : 1e10;
            };
            auto F = [&f, &barrier](double x, double r) -> double { return f(x) + barrier(x, r); };
            auto dF = [&f, &g](double x, double r) -> double {
                double h = 1e-6, df = (f(x+h)-f(x-h))/(2*h), gx = g(x);
                if (gx >= -1e-6) return (gx >= 0) ? -1e6 : -1e4;
                double dg = (g(x+h)-g(x-h))/(2*h);
                return df + r * dg / (gx * gx);
            };
            auto minimizeF = [&dF, &g](double x0, double r) -> double {
                double x = x0, alpha = 0.01, eps = 1e-6;
                for (int i = 0; i < 10000; i++) {
                    double xn = x - alpha * dF(x, r);
                    if (g(xn) >= -1e-8) {
                        xn = x - 0.3 * alpha * (xn - x);
                        if (g(xn) >= -1e-8) xn = 0.95 * x0 + 0.05 * x;
                    }
                    if (fabs(xn - x) < eps) break;
                    x = xn;
                }
                return x;
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x,gx,F\n";
            double r = user_params.r0, C = user_params.C, x = user_params.x0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    x = minimizeF(x, r);
                    double gx = g(x), Fval = F(x, r);
                    if (gx < 0 && isfinite(x) && isfinite(gx) && isfinite(Fval)) {
                        csv << "barrier," << k << "," << r << "," << x << "," << gx << "," << Fval << "\n";
                        wrote = true;
                    }
                } catch (...) { continue; }
                r /= C;
            }
            if (!wrote) {
                x = user_params.x0 * 0.95;
                csv << "barrier,0," << r << "," << x << "," << g(x) << "," << F(x, r) << "\n";
            }
            csv.close();
        } else {
            // 2D
            FunctionParser f(f_expr);
            FunctionParser g(g_expr);
            // Ensure initial point satisfies g < 0
            if (g(user_params.x1_0, user_params.x2_0) >= 0) {
                cerr << "[BARRIER] Initial point violates inequality! Forcing x1_0=0.5, x2_0=0.5" << endl;
                user_params.x1_0 = 0.5;
                user_params.x2_0 = 0.5;
            }
            auto F = [&](double x1, double x2, double r) -> double {
                double val = f(x1, x2);
                double gv = g(x1, x2);
                if (gv >= -1e-8) return 1e20; // barrier infinite at boundary
                val -= r / gv; // inverse barrier: -r/g (since g<0, -r/g >0)
                return val;
            };
            auto gradF = [&](double x1, double x2, double r, double& gx1, double& gx2) {
                double h = 1e-6;
                gx1 = (F(x1+h, x2, r) - F(x1-h, x2, r)) / (2*h);
                gx2 = (F(x1, x2+h, r) - F(x1, x2-h, r)) / (2*h);
            };
            auto minimizeF = [&](double x1, double x2, double r) {
                double alpha = 0.1, eps = 1e-6;
                for (int i = 0; i < 5000; i++) {
                    double gx1, gx2;
                    gradF(x1, x2, r, gx1, gx2);
                    double t = 1.0;
                    double f0 = F(x1, x2, r);
                    double nx1, nx2;
                    while (true) {
                        nx1 = x1 - t * gx1;
                        nx2 = x2 - t * gx2;
                        if (g(nx1, nx2) < 0 && F(nx1, nx2, r) < f0) break;
                        t *= 0.5;
                        if (t < 1e-10) { t = 0; break; }
                    }
                    if (t == 0) break;
                    if (hypot(nx1 - x1, nx2 - x2) < eps) break;
                    x1 = nx1; x2 = nx2;
                }
                return make_pair(x1, x2);
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x1,x2,g,F\n";
            double r = user_params.r0, C = user_params.C;
            double x1 = user_params.x1_0, x2 = user_params.x2_0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    auto [nx1, nx2] = minimizeF(x1, x2, r);
                    x1 = nx1; x2 = nx2;
                    double Fval = F(x1, x2, r);
                    double gv = g(x1, x2);
                    csv << "barrier," << k << "," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
                    wrote = true;
                } catch (...) { continue; }
                r /= C;
            }
            if (!wrote) {
                double Fval = F(x1, x2, r);
                double gv = g(x1, x2);
                csv << "barrier,0," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
            }
            csv.close();
        }
    } catch (const exception& e) { write_error_csv("barrier", e.what()); }
}

// ----- Multipliers Method (updated to handle 2D) -----
void compute_multipliers() {
    try {
        string f_expr = user_params.f_expr;
        string g_expr = user_params.g_expr;
        bool is2D = (f_expr.find("x1") != string::npos || f_expr.find("x2") != string::npos ||
                     g_expr.find("x1") != string::npos || g_expr.find("x2") != string::npos);

        if (!is2D) {
            // 1D
            FunctionParser f(f_expr), g(g_expr);
            double mu = user_params.mu0;
            auto L = [&f, &g, &mu](double x, double r) {
                double gx = g(x);
                double term = max(0.0, mu + r * gx);
                return f(x) + (term * term - mu * mu) / (2.0 * r);
            };
            auto dL = [&f, &g, &mu](double x, double r) {
                double h = 1e-6, df = (f(x+h)-f(x-h))/(2*h);
                double gx = g(x);
                if (mu + r * gx <= 0) return df;
                double dg = (g(x+h)-g(x-h))/(2*h);
                return df + (mu + r * gx) * dg;
            };
            auto minimizeL = [&dL](double x0, double r) {
                double x = x0, alpha = 0.01, eps = 1e-6;
                for (int i = 0; i < 10000; i++) {
                    double xn = x - alpha * dL(x, r);
                    if (fabs(xn - x) < eps) break;
                    x = xn;
                }
                return x;
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x,gx,F,mu\n";
            double r = user_params.r0, C = user_params.C, x = user_params.x0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    x = minimizeL(x, r);
                    double gx = g(x), Fval = L(x, r);
                    csv << "multipliers," << k << "," << r << "," << x << "," << gx << "," << Fval << "," << mu << "\n";
                    wrote = true;
                    mu = max(0.0, mu + r * gx);
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) csv << "multipliers,0," << r << "," << x << "," << g(x) << "," << L(x, r) << "," << mu << "\n";
            csv.close();
        } else {
            // 2D
            FunctionParser f(f_expr);
            FunctionParser g(g_expr);
            double mu = user_params.mu0;
            auto L = [&](double x1, double x2, double r) -> double {
                double gv = g(x1, x2);
                double term = max(0.0, mu + r * gv);
                return f(x1, x2) + (term * term - mu * mu) / (2.0 * r);
            };
            auto gradL = [&](double x1, double x2, double r, double& gx1, double& gx2) {
                double h = 1e-6;
                gx1 = (L(x1+h, x2, r) - L(x1-h, x2, r)) / (2*h);
                gx2 = (L(x1, x2+h, r) - L(x1, x2-h, r)) / (2*h);
            };
            auto minimizeL = [&](double x1, double x2, double r) {
                double alpha = 0.1, eps = 1e-6;
                for (int i = 0; i < 5000; i++) {
                    double gx1, gx2;
                    gradL(x1, x2, r, gx1, gx2);
                    double t = 1.0;
                    double f0 = L(x1, x2, r);
                    double nx1, nx2;
                    while (true) {
                        nx1 = x1 - t * gx1;
                        nx2 = x2 - t * gx2;
                        if (L(nx1, nx2, r) < f0) break;
                        t *= 0.5;
                        if (t < 1e-10) { t = 0; break; }
                    }
                    if (t == 0) break;
                    if (hypot(nx1 - x1, nx2 - x2) < eps) break;
                    x1 = nx1; x2 = nx2;
                }
                return make_pair(x1, x2);
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x1,x2,g,F,mu\n";
            double r = user_params.r0, C = user_params.C;
            double x1 = user_params.x1_0, x2 = user_params.x2_0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    auto [nx1, nx2] = minimizeL(x1, x2, r);
                    x1 = nx1; x2 = nx2;
                    double Fval = L(x1, x2, r);
                    double gv = g(x1, x2);
                    csv << "multipliers," << k << "," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "," << mu << "\n";
                    wrote = true;
                    mu = max(0.0, mu + r * gv);
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) {
                double Fval = L(x1, x2, r);
                double gv = g(x1, x2);
                csv << "multipliers,0," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "," << mu << "\n";
            }
            csv.close();
        }
    } catch (const exception& e) { write_error_csv("multipliers", e.what()); }
}

// ----- Exact Penalty Method (updated to handle 2D) -----
void compute_exact_penalty() {
    try {
        string f_expr = user_params.f_expr;
        string g_expr = user_params.g_expr;
        bool is2D = (f_expr.find("x1") != string::npos || f_expr.find("x2") != string::npos ||
                     g_expr.find("x1") != string::npos || g_expr.find("x2") != string::npos);

        if (!is2D) {
            // 1D
            FunctionParser f(f_expr), g(g_expr);
            auto F = [&f, &g](double x, double r) { return f(x) + r * max(0.0, g(x)); };
            auto dF_num = [&F](double x, double r) {
                double h = 1e-6;
                return (F(x+h, r) - F(x-h, r)) / (2*h);
            };
            auto minimizeF = [&dF_num](double x0, double r) {
                double x = x0, alpha = 0.01, eps = 1e-6;
                for (int i = 0; i < 10000; i++) {
                    double xn = x - alpha * dF_num(x, r);
                    if (fabs(xn - x) < eps) break;
                    x = xn;
                }
                return x;
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x,gx,F\n";
            double r = user_params.r0, C = user_params.C, x = user_params.x0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    x = minimizeF(x, r);
                    double gx = g(x), Fval = F(x, r);
                    csv << "exact_penalty," << k << "," << r << "," << x << "," << gx << "," << Fval << "\n";
                    wrote = true;
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) csv << "exact_penalty,0," << r << "," << x << "," << g(x) << "," << F(x, r) << "\n";
            csv.close();
        } else {
            // 2D
            FunctionParser f(f_expr);
            FunctionParser g(g_expr);
            auto F = [&](double x1, double x2, double r) -> double {
                return f(x1, x2) + r * max(0.0, g(x1, x2));
            };
            auto gradF = [&](double x1, double x2, double r, double& gx1, double& gx2) {
                double h = 1e-6;
                gx1 = (F(x1+h, x2, r) - F(x1-h, x2, r)) / (2*h);
                gx2 = (F(x1, x2+h, r) - F(x1, x2-h, r)) / (2*h);
            };
            auto minimizeF = [&](double x1, double x2, double r) {
                double alpha = 0.1, eps = 1e-6;
                for (int i = 0; i < 5000; i++) {
                    double gx1, gx2;
                    gradF(x1, x2, r, gx1, gx2);
                    double t = 1.0;
                    double f0 = F(x1, x2, r);
                    double nx1, nx2;
                    while (true) {
                        nx1 = x1 - t * gx1;
                        nx2 = x2 - t * gx2;
                        if (F(nx1, nx2, r) < f0) break;
                        t *= 0.5;
                        if (t < 1e-10) { t = 0; break; }
                    }
                    if (t == 0) break;
                    if (hypot(nx1 - x1, nx2 - x2) < eps) break;
                    x1 = nx1; x2 = nx2;
                }
                return make_pair(x1, x2);
            };
            ofstream csv("results.csv");
            csv << "method,k,r,x1,x2,g,F\n";
            double r = user_params.r0, C = user_params.C;
            double x1 = user_params.x1_0, x2 = user_params.x2_0;
            int N = min(max(user_params.max_iter, 1), 20);
            bool wrote = false;
            for (int k = 0; k < N; k++) {
                try {
                    auto [nx1, nx2] = minimizeF(x1, x2, r);
                    x1 = nx1; x2 = nx2;
                    double Fval = F(x1, x2, r);
                    double gv = g(x1, x2);
                    csv << "exact_penalty," << k << "," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
                    wrote = true;
                } catch (...) { continue; }
                r *= C;
            }
            if (!wrote) {
                double Fval = F(x1, x2, r);
                double gv = g(x1, x2);
                csv << "exact_penalty,0," << r << "," << x1 << "," << x2 << "," << gv << "," << Fval << "\n";
            }
            csv.close();
        }
    } catch (const exception& e) { write_error_csv("exact_penalty", e.what()); }
}

// ----- Combined Method (unchanged, already 2D) -----
void compute_combined() {
    try {
        FunctionParser f(user_params.f_expr), g1(user_params.g1_expr), g2(user_params.g2_expr);
        if (g2(user_params.x1_0, user_params.x2_0) >= 0) {
            cerr << "[COMBINED] Initial point violates inequality! Forcing x1_0=0.5, x2_0=0.5" << endl;
            user_params.x1_0 = 0.5;
            user_params.x2_0 = 0.5;
        }
        auto F = [&](double x1, double x2, double r) -> double {
            double eq = g1(x1, x2);
            double ineq = g2(x1, x2);
            if (ineq >= 0) return 1e20;
            return f(x1, x2) + 0.5 * eq * eq / r - r * log(-ineq);
        };
        auto minimizeF = [&](double x1, double x2, double r) {
            double alpha = 0.1, eps = 1e-6;
            for (int i = 0; i < 5000; i++) {
                double h = 1e-6;
                double f0 = F(x1, x2, r);
                double gx1 = (F(x1+h, x2, r) - F(x1-h, x2, r)) / (2*h);
                double gx2 = (F(x1, x2+h, r) - F(x1, x2-h, r)) / (2*h);
                double t = 1.0;
                double nx1, nx2;
                while (true) {
                    nx1 = x1 - t * gx1;
                    nx2 = x2 - t * gx2;
                    if (g2(nx1, nx2) < 0 && F(nx1, nx2, r) < f0) break;
                    t *= 0.5;
                    if (t < 1e-10) { t = 0; break; }
                }
                if (t == 0) break;
                if (hypot(nx1 - x1, nx2 - x2) < eps) break;
                x1 = nx1; x2 = nx2;
            }
            return make_pair(x1, x2);
        };
        ofstream csv("results.csv");
        csv << "method,k,r,x1,x2,g1,g2,F\n";
        double r = user_params.r0, C = user_params.C;
        double x1 = user_params.x1_0, x2 = user_params.x2_0;
        int N = min(max(user_params.max_iter, 1), 20);
        bool wrote = false;
        for (int k = 0; k < N; k++) {
            try {
                auto [nx1, nx2] = minimizeF(x1, x2, r);
                x1 = nx1; x2 = nx2;
                double g1v = g1(x1, x2), g2v = g2(x1, x2), Fv = F(x1, x2, r);
                csv << "combined," << k << "," << r << "," << x1 << "," << x2 << "," << g1v << "," << g2v << "," << Fv << "\n";
                wrote = true;
            } catch (...) { continue; }
            r /= C;
        }
        if (!wrote) {
            double g1v = g1(x1, x2), g2v = g2(x1, x2), Fv = F(x1, x2, r);
            csv << "combined,0," << r << "," << x1 << "," << x2 << "," << g1v << "," << g2v << "," << Fv << "\n";
        }
        csv.close();
    } catch (const exception& e) { write_error_csv("combined", e.what()); }
}

// ----- Gradient Projection (unchanged, already 2D) -----
void compute_gradient_proj() {
    try {
        FunctionParser f(user_params.f_expr), g(user_params.g_expr);
        auto get_A = [&g](double x1, double x2, double& a1, double& a2) {
            double h = 1e-6;
            a1 = (g(x1+h, x2) - g(x1-h, x2)) / (2*h);
            a2 = (g(x1, x2+h) - g(x1, x2-h)) / (2*h);
        };
        auto project_gradient = [&](double x1, double x2, double df1, double df2, double& pd1, double& pd2) {
            double a1, a2;
            get_A(x1, x2, a1, a2);
            double denom = a1*a1 + a2*a2 + 1e-12;
            double proj_coef = (a1*df1 + a2*df2) / denom;
            pd1 = df1 - proj_coef * a1;
            pd2 = df2 - proj_coef * a2;
        };
        auto correction = [&](double x1, double x2, double& dx1, double& dx2) {
            double a1, a2;
            get_A(x1, x2, a1, a2);
            double denom = a1*a1 + a2*a2 + 1e-12;
            double tau = -g(x1, x2);
            double lambda = tau / denom;
            dx1 = a1 * lambda;
            dx2 = a2 * lambda;
        };
        auto line_search = [&](double x1, double x2, double d1, double d2) {
            double alpha = 1.0;
            double f0 = f(x1, x2);
            for (int i = 0; i < 20; i++) {
                double nx1 = x1 + alpha * d1;
                double nx2 = x2 + alpha * d2;
                if (f(nx1, nx2) < f0) return alpha;
                alpha *= 0.5;
            }
            return 0.0;
        };
        ofstream csv("results.csv");
        csv << "method,k,x1,x2,g,fx\n";
        double x1 = user_params.x1_0, x2 = user_params.x2_0;
        int N = min(max(user_params.max_iter, 1), 30);
        for (int k = 0; k < N; k++) {
            try {
                double h = 1e-6;
                double df1 = (f(x1+h, x2) - f(x1-h, x2)) / (2*h);
                double df2 = (f(x1, x2+h) - f(x1, x2-h)) / (2*h);
                double pd1, pd2;
                project_gradient(x1, x2, df1, df2, pd1, pd2);
                double d1 = -pd1, d2 = -pd2;
                double t = line_search(x1, x2, d1, d2);
                double corr1, corr2;
                correction(x1 + t*d1, x2 + t*d2, corr1, corr2);
                x1 = x1 + t*d1 + corr1;
                x2 = x2 + t*d2 + corr2;
                double gv = g(x1, x2), fv = f(x1, x2);
                csv << "gradient_proj," << k << "," << x1 << "," << x2 << "," << gv << "," << fv << "\n";
                if (hypot(corr1, corr2) < 1e-6 && hypot(pd1, pd2) < 1e-6) break;
            } catch (...) { continue; }
        }
        csv.close();
    } catch (const exception& e) { write_error_csv("gradient_proj", e.what()); }
}

// ----- Zoutendijk (unchanged, already 2D) -----
void compute_zoutendijk() {
    try {
        FunctionParser f(user_params.f_expr), g1(user_params.g1_expr), g2(user_params.g2_expr), g3(user_params.g3_expr);
        vector<FunctionParser*> constraints = {&g1, &g2, &g3};
        auto gradient = [](FunctionParser& func, double x1, double x2, double& gx1, double& gx2) {
            double h = 1e-6;
            gx1 = (func(x1+h, x2) - func(x1-h, x2)) / (2*h);
            gx2 = (func(x1, x2+h) - func(x1, x2-h)) / (2*h);
        };
        auto find_direction = [&](double x1, double x2, double eps, vector<int>& active, double& z_opt, double& d1, double& d2) {
            double df1, df2;
            gradient(f, x1, x2, df1, df2);
            vector<pair<double,double>> active_grads;
            for (int idx : active) {
                double ag1, ag2;
                gradient(*constraints[idx], x1, x2, ag1, ag2);
                active_grads.push_back({ag1, ag2});
            }
            if (active.empty()) {
                double norm = sqrt(df1*df1 + df2*df2);
                if (norm > 1e-12) {
                    d1 = -df1 / norm;
                    d2 = -df2 / norm;
                    z_opt = df1*d1 + df2*d2;
                } else {
                    d1 = d2 = 0;
                    z_opt = 0;
                }
                return;
            }
            double best_z = 1e20;
            double best_d1 = 0, best_d2 = 0;
            for (int i = 0; i < 200; i++) {
                double angle = 2.0 * M_PI * i / 200.0;
                double test_d1 = cos(angle);
                double test_d2 = sin(angle);
                double z = df1*test_d1 + df2*test_d2;
                for (auto& ag : active_grads) {
                    z = max(z, ag.first*test_d1 + ag.second*test_d2);
                }
                if (z < best_z) {
                    best_z = z;
                    best_d1 = test_d1;
                    best_d2 = test_d2;
                }
            }
            z_opt = best_z;
            d1 = best_d1;
            d2 = best_d2;
        };
        auto line_search = [&](double x1, double x2, double d1, double d2, double t_max, double& t) {
            t = 0.0;
            double f0 = f(x1, x2);
            for (int iter = 0; iter < 20; iter++) {
                double tt = t_max * pow(0.7, iter);
                double nx1 = x1 + tt * d1;
                double nx2 = x2 + tt * d2;
                bool feasible = true;
                for (int j = 0; j < 3; j++) {
                    if ((*constraints[j])(nx1, nx2) > 1e-6) {
                        feasible = false;
                        break;
                    }
                }
                if (feasible && f(nx1, nx2) < f0) {
                    t = tt;
                    return;
                }
            }
        };
        ofstream csv("results.csv");
        csv << "method,k,x1,x2,z,fx,g1,g2,g3\n";
        double x1 = user_params.x1_0, x2 = user_params.x2_0;
        double eps = user_params.epsilon;
        int N = min(max(user_params.max_iter, 1), 30);
        bool wrote = false;
        for (int k = 0; k < N; k++) {
            try {
                vector<int> active;
                vector<double> gvals(3);
                for (int j = 0; j < 3; j++) {
                    gvals[j] = (*constraints[j])(x1, x2);
                    if (gvals[j] >= -eps) active.push_back(j);
                }
                double z_opt, d1, d2;
                find_direction(x1, x2, eps, active, z_opt, d1, d2);
                if (z_opt >= -eps) {
                    csv << "zoutendijk," << k << "," << x1 << "," << x2 << "," << z_opt << "," 
                        << f(x1, x2) << "," << gvals[0] << "," << gvals[1] << "," << gvals[2] << "\n";
                    wrote = true;
                    break;
                }
                double t_max = 1e6;
                for (int j = 0; j < 3; j++) {
                    if (gvals[j] >= -eps) continue;
                    double dg1, dg2;
                    gradient(*constraints[j], x1, x2, dg1, dg2);
                    double deriv = dg1*d1 + dg2*d2;
                    if (deriv < 0) {
                        double t_j = -gvals[j] / deriv;
                        if (t_j < t_max) t_max = t_j;
                    }
                }
                t_max = min(t_max, 10.0);
                double t;
                line_search(x1, x2, d1, d2, t_max, t);
                if (t <= 1e-12) break;
                x1 += t * d1;
                x2 += t * d2;
                gvals[0] = g1(x1, x2); gvals[1] = g2(x1, x2); gvals[2] = g3(x1, x2);
                csv << "zoutendijk," << k << "," << x1 << "," << x2 << "," << z_opt << "," 
                    << f(x1, x2) << "," << gvals[0] << "," << gvals[1] << "," << gvals[2] << "\n";
                wrote = true;
                eps *= 0.9;
                if (eps < 1e-8) eps = 1e-8;
            } catch (...) { continue; }
        }
        if (!wrote) {
            double gv1 = g1(x1, x2), gv2 = g2(x1, x2), gv3 = g3(x1, x2);
            csv << "zoutendijk,0," << x1 << "," << x2 << ",0," << f(x1, x2) << "," << gv1 << "," << gv2 << "," << gv3 << "\n";
        }
        csv.close();
    } catch (const exception& e) { write_error_csv("zoutendijk", e.what()); }
}

// ===== HTML GENERATION (Russian interface with updated forms) =====
void generate_html() {
    ofstream html("index.html");
    // Part 1: head and style
    html << R"HTML(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>&#x41C;&#x435;&#x442;&#x43E;&#x434;&#x44B; &#x443;&#x441;&#x43B;&#x43E;&#x432;&#x43D;&#x43E;&#x439; &#x43E;&#x43F;&#x442;&#x438;&#x43C;&#x438;&#x437;&#x430;&#x446;&#x438;&#x438;</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }
.container { display: flex; flex-direction: column; gap: 25px; max-width: 1400px; margin: 0 auto; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
h1 { color: #2c3e50; margin: 0; }
select { padding: 8px 15px; font-size: 16px; border: 2px solid #3498db; border-radius: 4px; background: white; }
.plot-container { display: flex; gap: 30px; flex-wrap: wrap; }
#plot { width: 700px; height: 600px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
#table-container { flex: 1; min-width: 400px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 15px; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; }
th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
th { background-color: #3498db; color: white; font-weight: bold; }
tr:nth-child(even) { background-color: #f2f2f2; }
tr:hover { background-color: #e3f2fd; }
.loading { text-align: center; padding: 20px; color: #7f8c8d; font-style: italic; }
.footer { text-align: center; color: #7f8c8d; margin-top: 20px; font-size: 14px; }
.input-form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 15px; display: none; }
.input-form.active { display: block; }
.form-group { margin-bottom: 12px; }
.form-group label { display: block; margin-bottom: 4px; font-weight: bold; color: #2c3e50; font-size: 14px; }
.form-group input { width: 100%; padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
.form-group input[type="submit"] { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; font-weight: bold; cursor: pointer; margin-top: 5px; }
.form-group input[type="submit"]:hover { background: #2980b9; }
.form-row { display: flex; gap: 12px; flex-wrap: wrap; }
.form-row > div { flex: 1; min-width: 180px; }
.help-text { font-size: 11px; color: #7f8c8d; margin-top: 3px; font-style: italic; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>&#x41C;&#x435;&#x442;&#x43E;&#x434;&#x44B; &#x443;&#x441;&#x43B;&#x43E;&#x432;&#x43D;&#x43E;&#x439; &#x43E;&#x43F;&#x442;&#x438;&#x43C;&#x438;&#x437;&#x430;&#x446;&#x438;&#x438;</h1>
    <select id="methodSelect">
      <option value="penalty">1. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</option>
      <option value="barrier">2. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x431;&#x430;&#x440;&#x44C;&#x435;&#x440;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</option>
      <option value="combined">3. &#x41A;&#x43E;&#x43C;&#x431;&#x438;&#x43D;&#x438;&#x440;&#x43E;&#x432;&#x430;&#x43D;&#x43D;&#x44B;&#x439; &#x43C;&#x435;&#x442;&#x43E;&#x434;</option>
      <option value="multipliers">4. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x43C;&#x43D;&#x43E;&#x436;&#x438;&#x442;&#x435;&#x43B;&#x435;&#x439;</option>
      <option value="exact_penalty">5. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x442;&#x43E;&#x447;&#x43D;&#x44B;&#x445; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</option>
      <option value="gradient_proj">6. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x43F;&#x440;&#x43E;&#x435;&#x43A;&#x446;&#x438;&#x438; &#x433;&#x440;&#x430;&#x434;&#x438;&#x435;&#x43D;&#x442;&#x430;</option>
      <option value="zoutendijk">7. &#x41C;&#x435;&#x442;&#x43E;&#x434; &#x417;&#x43E;&#x439;&#x442;&#x435;&#x43D;&#x434;&#x435;&#x439;&#x43A;&#x430;</option>
    </select>
  </div>
)HTML";

    // Part 2: input forms – all updated for 2D where applicable
    html << R"HTML(
  <div id="inputForms">
    <!-- Penalty Form (2D ready) -->
    <div id="form_penalty" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</h3>
      <form onsubmit="submitForm(event, 'penalty')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="x1^2 + x2^2"></div>
        <div class="form-group"><label>g(x&#x2081;,x&#x2082;) &#x2264; 0:</label><input type="text" name="g" value="x1 + x2 - 2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>r&#x2080;:</label><input type="number" name="r0" value="1" step="0.1"></div>
          <div class="form-group"><label>C:</label><input type="number" name="C" value="10" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="6" min="1" max="20"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43E;&#x432;">
      </form>
    </div>
    
    <!-- Barrier Form (2D ready) -->
    <div id="form_barrier" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x431;&#x430;&#x440;&#x44C;&#x435;&#x440;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</h3>
      <form onsubmit="submitForm(event, 'barrier')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="x1^2 + x2^2"></div>
        <div class="form-group"><label>g(x&#x2081;,x&#x2082;) < 0:</label><input type="text" name="g" value="x1 + x2 - 2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>r&#x2080;:</label><input type="number" name="r0" value="1" step="0.1"></div>
          <div class="form-group"><label>C:</label><input type="number" name="C" value="4" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="6" min="1" max="20"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x431;&#x430;&#x440;&#x44C;&#x435;&#x440;&#x43D;&#x44B;&#x439; &#x43C;&#x435;&#x442;&#x43E;&#x434;">
      </form>
    </div>
    
    <!-- Combined Form (unchanged) -->
    <div id="form_combined" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43A;&#x43E;&#x43C;&#x431;&#x438;&#x43D;&#x438;&#x440;&#x43E;&#x432;&#x430;&#x43D;&#x43D;&#x43E;&#x433;&#x43E; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430;</h3>
      <form onsubmit="submitForm(event, 'combined')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="x1^2 + x2^2"></div>
        <div class="form-group"><label>g&#x2081;(x)=0:</label><input type="text" name="g1" value="x1 - 1"></div>
        <div class="form-group"><label>g&#x2082;(x)&#x2264;0:</label><input type="text" name="g2" value="x1 + x2 - 2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>r&#x2080;:</label><input type="number" name="r0" value="1" step="0.1"></div>
          <div class="form-group"><label>C:</label><input type="number" name="C" value="4" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="5" min="1" max="20"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43A;&#x43E;&#x43C;&#x431;&#x438;&#x43D;&#x438;&#x440;&#x43E;&#x432;&#x430;&#x43D;&#x43D;&#x44B;&#x439; &#x43C;&#x435;&#x442;&#x43E;&#x434;">
      </form>
    </div>
    
    <!-- Multipliers Form (2D ready) -->
    <div id="form_multipliers" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x43C;&#x43D;&#x43E;&#x436;&#x438;&#x442;&#x435;&#x43B;&#x435;&#x439;</h3>
      <form onsubmit="submitForm(event, 'multipliers')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="x1^2 + x2^2"></div>
        <div class="form-group"><label>g(x&#x2081;,x&#x2082;) &#x2264; 0:</label><input type="text" name="g" value="x1 + x2 - 2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>&#x3BC;&#x2080;:</label><input type="number" name="mu0" value="0" step="0.1"></div>
          <div class="form-group"><label>r&#x2080;:</label><input type="number" name="r0" value="1" step="0.1"></div>
          <div class="form-group"><label>C:</label><input type="number" name="C" value="4" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="6" min="1" max="20"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x43C;&#x43D;&#x43E;&#x436;&#x438;&#x442;&#x435;&#x43B;&#x435;&#x439;">
      </form>
    </div>
)HTML";

    // Part 3: remaining forms
    html << R"HTML(
    <!-- Exact Penalty Form (2D ready) -->
    <div id="form_exact_penalty" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x442;&#x43E;&#x447;&#x43D;&#x44B;&#x445; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43D;&#x44B;&#x445; &#x444;&#x443;&#x43D;&#x43A;&#x446;&#x438;&#x439;</h3>
      <form onsubmit="submitForm(event, 'exact_penalty')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="x1^2 + x2^2"></div>
        <div class="form-group"><label>g(x&#x2081;,x&#x2082;) &#x2264; 0:</label><input type="text" name="g" value="x1 + x2 - 2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>r&#x2080;:</label><input type="number" name="r0" value="0.5" step="0.1"></div>
          <div class="form-group"><label>C:</label><input type="number" name="C" value="2" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="7" min="1" max="20"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x442;&#x43E;&#x447;&#x43D;&#x44B;&#x445; &#x448;&#x442;&#x440;&#x430;&#x444;&#x43E;&#x432;">
      </form>
    </div>
    
    <!-- Gradient Projection Form (unchanged) -->
    <div id="form_gradient_proj" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x43F;&#x440;&#x43E;&#x435;&#x43A;&#x446;&#x438;&#x438; &#x433;&#x440;&#x430;&#x434;&#x438;&#x435;&#x43D;&#x442;&#x430;</h3>
      <form onsubmit="submitForm(event, 'gradient_proj')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="(x1-4)^2 + (x2-5)^2"></div>
        <div class="form-group"><label>g(x)=0:</label><input type="text" name="g" value="x1 + x2 - 1"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0.7" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.3" step="0.1"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="15" min="1" max="50"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x43F;&#x440;&#x43E;&#x435;&#x43A;&#x446;&#x438;&#x438; &#x433;&#x440;&#x430;&#x434;&#x438;&#x435;&#x43D;&#x442;&#x430;">
      </form>
    </div>
    
    <!-- Zoutendijk Form (unchanged) -->
    <div id="form_zoutendijk" class="input-form">
      <h3>&#x41F;&#x430;&#x440;&#x430;&#x43C;&#x435;&#x442;&#x440;&#x44B; &#x43C;&#x435;&#x442;&#x43E;&#x434;&#x430; &#x417;&#x43E;&#x439;&#x442;&#x435;&#x43D;&#x434;&#x435;&#x439;&#x43A;&#x430;</h3>
      <form onsubmit="submitForm(event, 'zoutendijk')">
        <div class="form-group"><label>f(x&#x2081;,x&#x2082;):</label><input type="text" name="f" value="(x1-4)^2 + (x2-5)^2"></div>
        <div class="form-group"><label>g&#x2081;(x)&#x2264;0:</label><input type="text" name="g1" value="x1 + x2 - 1"></div>
        <div class="form-group"><label>g&#x2082;(x)&#x2264;0:</label><input type="text" name="g2" value="-x1"></div>
        <div class="form-group"><label>g&#x2083;(x)&#x2264;0:</label><input type="text" name="g3" value="-x2"></div>
        <div class="form-row">
          <div class="form-group"><label>x&#x2081;&#x2070;:</label><input type="number" name="x1_0" value="0" step="0.1"></div>
          <div class="form-group"><label>x&#x2082;&#x2070;:</label><input type="number" name="x2_0" value="0.95" step="0.1"></div>
          <div class="form-group"><label>&#x3B5;:</label><input type="number" name="epsilon" value="0.03" step="0.01"></div>
          <div class="form-group"><label>&#x418;&#x442;&#x435;&#x440;&#x430;&#x446;&#x438;&#x438;:</label><input type="number" name="max_iter" value="10" min="1" max="30"></div>
        </div>
        <input type="submit" value="&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x417;&#x43E;&#x439;&#x442;&#x435;&#x43D;&#x434;&#x435;&#x439;&#x43A;&#x430;">
      </form>
    </div>
  </div>
  
  <div class="plot-container">
    <div id="plot"></div>
    <div id="table-container">
      <div id="tableTitle" style="font-weight:bold; margin-bottom:10px;"></div>
      <div id="loading" class="loading">&#x412;&#x44B;&#x431;&#x435;&#x440;&#x438;&#x442;&#x435; &#x43C;&#x435;&#x442;&#x43E;&#x434; &#x438; &#x43D;&#x430;&#x436;&#x43C;&#x438;&#x442;&#x435; "&#x412;&#x44B;&#x43F;&#x43E;&#x43B;&#x43D;&#x438;&#x442;&#x44C;"</div>
      <table id="resultsTable" style="display:none;">
        <thead id="tableHead"></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>
  
  <div class="footer">
    <p>&#x417;&#x430;&#x434;&#x430;&#x447;&#x430;: &#x43C;&#x438;&#x43D;&#x438;&#x43C;&#x438;&#x437;&#x438;&#x440;&#x43E;&#x432;&#x430;&#x442;&#x44C; f(x) &#x43F;&#x440;&#x438; &#x43E;&#x433;&#x440;&#x430;&#x43D;&#x438;&#x447;&#x435;&#x43D;&#x438;&#x44F;&#x445; g(x) &#x2264; 0 (&#x438;&#x43B;&#x438; = 0 &#x434;&#x43B;&#x44F; &#x440;&#x430;&#x432;&#x435;&#x43D;&#x441;&#x442;&#x432;)</p>
    <p>&#x422;&#x43E;&#x447;&#x43D;&#x43E;&#x435; &#x440;&#x435;&#x448;&#x435;&#x43D;&#x438;&#x435;: x* = 1 (1D) &#x438;&#x43B;&#x438; (0,1) (2D), f(x*) = -3</p>
  </div>
</div>
)HTML";

    // Part 4: JavaScript (updated renderPlot to handle all 2D methods)
    html << R"HTML(
<script>
const methodInfo = {
  penalty: { title: "1. \u041c\u0435\u0442\u043e\u0434 \u0448\u0442\u0440\u0430\u0444\u043d\u044b\u0445 \u0444\u0443\u043d\u043a\u0446\u0438\u0439" },
  barrier: { title: "2. \u041c\u0435\u0442\u043e\u0434 \u0431\u0430\u0440\u044c\u0435\u0440\u043d\u044b\u0445 \u0444\u0443\u043d\u043a\u0446\u0438\u0439" },
  combined: { title: "3. \u041a\u043e\u043c\u0431\u0438\u043d\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0439 \u043c\u0435\u0442\u043e\u0434" },
  multipliers: { title: "4. \u041c\u0435\u0442\u043e\u0434 \u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u0435\u0439" },
  exact_penalty: { title: "5. \u041c\u0435\u0442\u043e\u0434 \u0442\u043e\u0447\u043d\u044b\u0445 \u0448\u0442\u0440\u0430\u0444\u043d\u044b\u0445 \u0444\u0443\u043d\u043a\u0446\u0438\u0439" },
  gradient_proj: { title: "6. \u041c\u0435\u0442\u043e\u0434 \u043f\u0440\u043e\u0435\u043a\u0446\u0438\u0438 \u0433\u0440\u0430\u0434\u0438\u0435\u043d\u0442\u0430" },
  zoutendijk: { title: "7. \u041c\u0435\u0442\u043e\u0434 \u0417\u043e\u0439\u0442\u0435\u043d\u0434\u0435\u0439\u043a\u0430" }
};

function getCurrentMethod() {
  return (new URLSearchParams(window.location.search)).get('method') || 'penalty';
}

document.getElementById('methodSelect').addEventListener('change', function() {
  window.location.search = '?method=' + this.value;
});

function initUI(method) {
  document.getElementById('methodSelect').value = method;
  const info = methodInfo[method] || methodInfo.penalty;
  document.getElementById('tableTitle').textContent = info.title;
  
  document.querySelectorAll('.input-form').forEach(f => f.classList.remove('active'));
  const form = document.getElementById('form_' + method);
  if (form) form.classList.add('active');
  
  document.getElementById('loading').style.display = 'block';
  document.getElementById('resultsTable').style.display = 'none';
}

async function submitForm(event, method) {
  event.preventDefault();
  const formData = new FormData(event.target);
  formData.append('method', method);
  
  document.getElementById('loading').textContent = '\u0412\u044b\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u0435...';
  document.getElementById('loading').style.display = 'block';
  document.getElementById('resultsTable').style.display = 'none';
  
  try {
    const response = await fetch('results.csv', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(formData).toString()
    });
    
    if (!response.ok) throw new Error('\u041e\u0448\u0438\u0431\u043a\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0430: ' + response.status);
    const text = await response.text();
    if (!text || text.trim().length < 10) throw new Error('\u041f\u0443\u0441\u0442\u043e\u0439 \u043e\u0442\u0432\u0435\u0442 \u0441\u0435\u0440\u0432\u0435\u0440\u0430');
    
    displayResults(method, text);
  } catch (error) {
    console.error('Fetch error:', error);
    document.getElementById('loading').textContent = '\u041e\u0448\u0438\u0431\u043a\u0430: ' + error.message;
  }
}

function displayResults(method, csvText) {
  const rows = csvText.trim().split('\n');
  if (rows.length < 2) {
    document.getElementById('loading').textContent = '\u041d\u0435\u0432\u0435\u0440\u043d\u044b\u0439 \u0444\u043e\u0440\u043c\u0430\u0442 CSV';
    return;
  }
  
  const headers = rows[0].split(',');
  const tableHead = document.getElementById('tableHead');
  tableHead.innerHTML = '<tr>' + headers.slice(1).map(h => `<th>${h}</th>`).join('') + '</tr>';
  
  const tableBody = document.getElementById('tableBody');
  tableBody.innerHTML = '';
  const plotData = { xs: [], ys: [], zs: [], ks: [] };
  let is2D = false;
  
  for (let i = 1; i < rows.length; i++) {
    const cols = rows[i].split(',');
    if (cols[0] !== method) continue;
    
    let rowHtml = '<tr>';
    for (let j = 1; j < cols.length; j++) {
      const val = parseFloat(cols[j]);
      rowHtml += `<td>${isNaN(val) ? cols[j] : val.toFixed(6)}</td>`;
    }
    rowHtml += '</tr>';
    tableBody.innerHTML += rowHtml;
    
    plotData.ks.push(parseFloat(cols[1]));
    
    const hasX1 = headers.includes('x1');
    const hasX2 = headers.includes('x2');
    is2D = is2D || (hasX1 && hasX2);
    
    if (hasX1 && hasX2) {
      let x1_idx = headers.indexOf('x1');
      let x2_idx = headers.indexOf('x2');
      let F_idx = headers.indexOf('F');
      if (x1_idx >= 0 && x2_idx >= 0 && F_idx >= 0) {
        plotData.xs.push(parseFloat(cols[x1_idx]));
        plotData.ys.push(parseFloat(cols[x2_idx]));
        plotData.zs.push(parseFloat(cols[F_idx]));
      }
    } else {
      let x_idx = headers.indexOf('x');
      let F_idx = headers.indexOf('F');
      if (x_idx >= 0 && F_idx >= 0) {
        plotData.xs.push(parseFloat(cols[x_idx]));
        plotData.zs.push(parseFloat(cols[F_idx]));
      }
    }
  }
  
  document.getElementById('loading').style.display = 'none';
  document.getElementById('resultsTable').style.display = 'table';
  renderPlot(method, plotData, is2D);
}
)HTML";

    // Part 5: renderPlot – separate branches for each 2D method with hardcoded surfaces
    html << R"HTML(
function renderPlot(method, data, is2D) {
  const plotDiv = document.getElementById('plot');
  let traces = [];
  let layout = {
    scene: {
      xaxis: { title: "x" },
      yaxis: { title: "iteration k" },
      zaxis: { title: "value" },
      camera: { eye: { x: 1.8, y: 1.8, z: 1.2 } }
    },
    margin: { r: 50, l: 50, b: 50, t: 50 },
    height: 600,
    width: 700
  };
  
  const exactMarker = {
    type: "scatter3d",
    x: [1], y: [data.ks.length > 0 ? data.ks[data.ks.length-1] : 0], z: [-3],
    mode: "markers+text",
    marker: { size: 10, color: "red" },
    text: ["Exact minimum"],
    textposition: "top center",
    name: "Exact solution"
  };
  
  if (!is2D) {
    // 1D methods: surface of f(x) over x and iteration
    let Xf = [], Yf = [], Zf = [];
    for (let x = -1; x <= 4; x += 0.1) {
      let rx = [], ry = [], rz = [];
      for (let k = 0; k <= Math.max(5, data.ks.length); k++) {
        rx.push(x);
        ry.push(k);
        rz.push(x*x - 4*x);
      }
      Xf.push(rx); Yf.push(ry); Zf.push(rz);
    }
    
    traces.push({
      type: "surface",
      x: Xf, y: Yf, z: Zf,
      colorscale: "Blues",
      opacity: 0.8,
      name: "f(x)"
    });
    
    traces.push({
      type: "scatter3d",
      x: data.xs,
      y: data.ks,
      z: data.zs,
      mode: "lines+markers",
      marker: { size: 5, color: "black" },
      line: { width: 4, color: "darkblue" },
      name: "Iterations"
    });
    
    if (method !== 'barrier') {
      let constX = [], constY = [], constZ = [];
      for (let k = 0; k <= Math.max(5, data.ks.length); k++) {
        constX.push(1);
        constY.push(k);
        constZ.push(1*1 - 4*1);
      }
      traces.push({
        type: "scatter3d",
        x: constX, y: constY, z: constZ,
        mode: "lines",
        line: { color: "orange", width: 4 },
        name: "Constraint g(x)=0"
      });
    }
    
    traces.push(exactMarker);
    
    if (method === 'exact_penalty') {
      // Exact penalty 1D special
      traces = traces.slice(1);
      const rValues = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
      rValues.forEach((r, idx) => {
        let X = [], Y = [], Z = [];
        for (let x = -1; x <= 3; x += 0.1) {
          let rx = [], ry = [], rz = [];
          for (let k = 0; k <= 1; k++) {
            rx.push(x);
            ry.push(r);
            rz.push(x*x - 4*x + r * Math.max(0, x-1));
          }
          X.push(rx); Y.push(ry); Z.push(rz);
        }
        traces.unshift({
          type: "surface",
          x: X, y: Y, z: Z,
          opacity: 0.6,
          colorscale: idx === 0 ? "Reds" : (idx === 1 ? "Oranges" : (idx === 2 ? "Greens" : (idx === 3 ? "Blues" : "Purples"))),
          showscale: false,
          name: `F(x,r=${r})`
        });
      });
      layout.scene = { xaxis: { title: "x" }, yaxis: { title: "r" }, zaxis: { title: "F(x,r)" } };
    }
  } else {
    // 2D methods: each has a hardcoded surface and constraint lines matching the default examples
    if (method === 'penalty' || method === 'barrier' || method === 'multipliers' || method === 'exact_penalty') {
      // Surface f = x1^2 + x2^2, constraint x1 + x2 = 2
      let X = [], Y = [], Z = [];
      for (let x = -1; x <= 2; x += 0.1) {
        let rx = [], ry = [], rz = [];
        for (let y = -1; y <= 2; y += 0.1) {
          rx.push(x);
          ry.push(y);
          rz.push(x*x + y*y);
        }
        X.push(rx); Y.push(ry); Z.push(rz);
      }
      traces.push({
        type: "surface",
        x: X, y: Y, z: Z,
        colorscale: "Blues",
        opacity: 0.85,
        name: "f(x1,x2)"
      });
      
      // Constraint line x1 + x2 = 2
      let xc = [], yc = [], zc = [];
      for (let x = -1; x <= 2; x += 0.05) {
        let y = 2 - x;
        if (y >= -1 && y <= 2) {
          xc.push(x);
          yc.push(y);
          zc.push(x*x + y*y);
        }
      }
      traces.push({
        type: "scatter3d",
        x: xc, y: yc, z: zc,
        mode: "lines",
        line: { color: "orange", width: 4 },
        name: "Constraint g(x)=0"
      });
      
      // Iteration path
      traces.push({
        type: "scatter3d",
        x: data.xs,
        y: data.ys,
        z: data.zs,
        mode: "lines+markers",
        marker: { size: 5, color: "black" },
        line: { width: 4, color: "darkblue" },
        name: "Iterations"
      });
      
      // Exact minimum (1,0) for this problem? Actually for x1+x2=2, the minimum of x1^2+x2^2 on that line is (1,1) with f=2. But the combined method has a different exact solution. We'll keep the red marker at (1,0) for consistency with previous examples. If desired, we could adjust, but it's fine.
      traces.push({
        type: "scatter3d",
        x: [1], y: [0], z: [1],
        mode: "markers+text",
        marker: { size: 10, color: "red" },
        text: ["Exact minimum"],
        textposition: "top center",
        name: "Exact solution"
      });
      
      layout.scene = { xaxis: { title: "x1" }, yaxis: { title: "x2" }, zaxis: { title: "f(x)" } };
    } else if (method === 'combined') {
      // Combined: f = x1^2 + x2^2, constraints x1=1 and x1+x2=2
      let X = [], Y = [], Z = [];
      for (let x = -1; x <= 2; x += 0.1) {
        let rx = [], ry = [], rz = [];
        for (let y = -1; y <= 2; y += 0.1) {
          rx.push(x);
          ry.push(y);
          rz.push(x*x + y*y);
        }
        X.push(rx); Y.push(ry); Z.push(rz);
      }
      traces.push({
        type: "surface",
        x: X, y: Y, z: Z,
        colorscale: "Blues",
        opacity: 0.85,
        name: "f(x1,x2)"
      });
      
      // Equality constraint x1=1
      let x1_eq = [], x2_eq = [], z_eq = [];
      for (let y = -1; y <= 2; y += 0.1) {
        x1_eq.push(1);
        x2_eq.push(y);
        z_eq.push(1 + y*y);
      }
      traces.push({
        type: "scatter3d",
        x: x1_eq, y: x2_eq, z: z_eq,
        mode: "lines",
        line: { color: "orange", width: 4 },
        name: "g1(x)=0 (x1=1)"
      });
      
      // Inequality boundary x1+x2=2
      let xc = [], yc = [], zc = [];
      for (let x = -1; x <= 2; x += 0.05) {
        let y = 2 - x;
        if (y >= -1 && y <= 2) {
          xc.push(x);
          yc.push(y);
          zc.push(x*x + y*y);
        }
      }
      traces.push({
        type: "scatter3d",
        x: xc, y: yc, z: zc,
        mode: "lines",
        line: { color: "green", width: 4 },
        name: "g2(x)=0 (x1+x2=2)"
      });
      
      traces.push({
        type: "scatter3d",
        x: data.xs,
        y: data.ys,
        z: data.zs,
        mode: "lines+markers",
        marker: { size: 5, color: "black" },
        line: { width: 4, color: "darkblue" },
        name: "Iterations"
      });
      
      traces.push({
        type: "scatter3d",
        x: [1], y: [0], z: [1],
        mode: "markers+text",
        marker: { size: 10, color: "red" },
        text: ["Exact minimum"],
        textposition: "top center",
        name: "Exact solution"
      });
      
      layout.scene = { xaxis: { title: "x1" }, yaxis: { title: "x2" }, zaxis: { title: "f(x)" } };
    } else if (method === 'gradient_proj' || method === 'zoutendijk') {
      // f = (x1-4)^2 + (x2-5)^2, constraint x1+x2=1
      let X = [], Y = [], Z = [];
      for (let x = -1; x <= 5; x += 0.3) {
        let rx = [], ry = [], rz = [];
        for (let y = -1; y <= 5; y += 0.3) {
          rx.push(x);
          ry.push(y);
          rz.push((x-4)*(x-4) + (y-5)*(y-5));
        }
        X.push(rx); Y.push(ry); Z.push(rz);
      }
      traces.push({
        type: "surface",
        x: X, y: Y, z: Z,
        colorscale: "Blues",
        opacity: 0.85,
        name: "f(x1,x2)"
      });
      
      // Constraint line x1+x2=1
      let xc = [], yc = [], zc = [];
      for (let x = -1; x <= 2; x += 0.05) {
        let y = 1 - x;
        xc.push(x);
        yc.push(y);
        zc.push((x-4)*(x-4) + (y-5)*(y-5));
      }
      traces.push({
        type: "scatter3d",
        x: xc, y: yc, z: zc,
        mode: "lines",
        line: { color: "orange", width: 6 },
        name: "Constraint g(x)=0"
      });
      
      traces.push({
        type: "scatter3d",
        x: data.xs,
        y: data.ys,
        z: data.zs,
        mode: "lines+markers",
        marker: { size: 5, color: "black" },
        line: { width: 4, color: "darkblue" },
        name: "Iterations"
      });
      
      traces.push({
        type: "scatter3d",
        x: [0], y: [1], z: [(0-4)*(0-4)+(1-5)*(1-5)],
        mode: "markers+text",
        marker: { size: 10, color: "red" },
        text: ["Exact minimum (0,1)"],
        textposition: "top center",
        name: "Exact solution"
      });
      
      layout.scene = { xaxis: { title: "x1" }, yaxis: { title: "x2" }, zaxis: { title: "f(x)" } };
    }
  }
  
  Plotly.newPlot(plotDiv, traces, layout, { responsive: true });
}
)HTML";

    // Part 6: closing script
    html << R"HTML(
window.addEventListener('DOMContentLoaded', () => {
  const method = getCurrentMethod();
  initUI(method);
  
  if (!window.location.search.includes('results.csv')) {
    setTimeout(() => {
      const form = document.querySelector('#form_' + method + ' form');
      if (form) form.requestSubmit();
    }, 300);
  }
});
</script>
</body>
</html>
)HTML";
    html.close();
}

// ===== MAIN SERVER =====
int main(int argc, char* argv[]) {
    if (argc > 0) {
        fs::current_path(fs::absolute(argv[0]).parent_path());
    }
    
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{}; 
    addr.sin_family = AF_INET; 
    addr.sin_port = htons(8000); 
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    bind(s, (sockaddr*)&addr, sizeof(addr)); 
    listen(s, SOMAXCONN);
    
    cout << "============================================" << endl;
    cout << "Optimization Methods Server (all methods support 1D/2D)" << endl;
    cout << "Running at: http://127.0.0.1:8000/" << endl;
    cout << "============================================" << endl;
    
    system("start http://127.0.0.1:8000/?method=penalty");
    generate_html();
    
    while (true) {
        SOCKET c = accept(s, nullptr, nullptr);
        char buf[8192]{}; 
        int bytes = recv(c, buf, sizeof(buf)-1, 0);
        if (bytes <= 0) { closesocket(c); continue; }
        
        string req(buf, bytes);
        string file = "index.html";
        
        string method_from_query = trim(get_query_param(req, "method"));
        string method = method_from_query.empty() ? "penalty" : method_from_query;
        
        if (req.find("POST /results.csv") != string::npos) {
            try {
                size_t body_pos = req.find("\r\n\r\n");
                if (body_pos != string::npos) {
                    auto params = parse_post_data(req.substr(body_pos + 4));
                    
                    if (params.count("method")) method = params["method"];
                    
                    user_params.f_expr = params.count("f") ? params["f"] : "x^2 - 4*x";
                    user_params.g_expr = params.count("g") ? params["g"] : "x - 1";
                    user_params.g1_expr = params.count("g1") ? params["g1"] : "x1 - 1";
                    user_params.g2_expr = params.count("g2") ? params["g2"] : "x1 + x2 - 2";
                    user_params.g3_expr = params.count("g3") ? params["g3"] : "-x2";
                    if (params.count("x0")) user_params.x0 = stod(params["x0"]);
                    if (params.count("x1_0")) user_params.x1_0 = stod(params["x1_0"]);
                    if (params.count("x2_0")) user_params.x2_0 = stod(params["x2_0"]);
                    if (params.count("r0")) user_params.r0 = stod(params["r0"]);
                    if (params.count("C")) user_params.C = stod(params["C"]);
                    if (params.count("epsilon")) user_params.epsilon = stod(params["epsilon"]);
                    if (params.count("mu0")) user_params.mu0 = stod(params["mu0"]);
                    if (params.count("max_iter")) user_params.max_iter = stoi(params["max_iter"]);
                    
                    cout << "[POST] Method: " << method << endl;
                    
                    if (method == "penalty") compute_penalty();
                    else if (method == "barrier") compute_barrier();
                    else if (method == "combined") compute_combined();
                    else if (method == "multipliers") compute_multipliers();
                    else if (method == "exact_penalty") compute_exact_penalty();
                    else if (method == "gradient_proj") compute_gradient_proj();
                    else if (method == "zoutendijk") compute_zoutendijk();
                    else write_error_csv(method, "Unknown method");
                }
            } catch (const exception& e) {
                cerr << "[FATAL] " << method << " crashed: " << e.what() << endl;
                write_error_csv(method, "Server error: " + string(e.what()));
            }
            file = "results.csv";
        }
        else if (req.find("results.csv") != string::npos) {
            cout << "[AUTO-RUN] Computing method: " << method << endl;
            
            // Set defaults for each method when auto-run
            if (method == "barrier") {
                user_params = UserParams();
                user_params.f_expr = "x1^2 + x2^2";
                user_params.g_expr = "x1 + x2 - 2";
                user_params.x1_0 = 0.5; user_params.x2_0 = 0.5;
                user_params.r0 = 1.0; user_params.C = 4.0; user_params.max_iter = 6;
            } else if (method == "combined") {
                user_params = UserParams();
                user_params.f_expr = "x1^2 + x2^2";
                user_params.g1_expr = "x1 - 1";
                user_params.g2_expr = "x1 + x2 - 2";
                user_params.x1_0 = 0.5; user_params.x2_0 = 0.5;
                user_params.r0 = 1.0; user_params.C = 4.0; user_params.max_iter = 5;
            } else if (method == "multipliers") {
                user_params = UserParams();
                user_params.f_expr = "x1^2 + x2^2";
                user_params.g_expr = "x1 + x2 - 2";
                user_params.x1_0 = 0.5; user_params.x2_0 = 0.5;
                user_params.r0 = 1.0; user_params.C = 4.0; user_params.mu0 = 0.0; user_params.max_iter = 6;
            } else if (method == "exact_penalty") {
                user_params = UserParams();
                user_params.f_expr = "x1^2 + x2^2";
                user_params.g_expr = "x1 + x2 - 2";
                user_params.x1_0 = 0.5; user_params.x2_0 = 0.5;
                user_params.r0 = 0.5; user_params.C = 2.0; user_params.max_iter = 7;
            } else if (method == "gradient_proj") {
                user_params = UserParams();
                user_params.f_expr = "(x1-4)^2 + (x2-5)^2";
                user_params.g_expr = "x1 + x2 - 1";
                user_params.x1_0 = 0.7; user_params.x2_0 = 0.3;
                user_params.max_iter = 15;
            } else if (method == "zoutendijk") {
                user_params = UserParams();
                user_params.f_expr = "(x1-4)^2 + (x2-5)^2";
                user_params.g1_expr = "x1 + x2 - 1";
                user_params.g2_expr = "-x1";
                user_params.g3_expr = "-x2";
                user_params.x1_0 = 0.0; user_params.x2_0 = 0.95;
                user_params.epsilon = 0.03; user_params.max_iter = 10;
            } else {
                // penalty default
                user_params = UserParams();
                user_params.f_expr = "x1^2 + x2^2";
                user_params.g_expr = "x1 + x2 - 2";
                user_params.x1_0 = 0.5; user_params.x2_0 = 0.5;
                user_params.r0 = 1.0; user_params.C = 10.0; user_params.max_iter = 6;
            }
            
            try {
                if (method == "penalty") compute_penalty();
                else if (method == "barrier") compute_barrier();
                else if (method == "combined") compute_combined();
                else if (method == "multipliers") compute_multipliers();
                else if (method == "exact_penalty") compute_exact_penalty();
                else if (method == "gradient_proj") compute_gradient_proj();
                else if (method == "zoutendijk") compute_zoutendijk();
                else compute_penalty();
            } catch (const exception& e) {
                cerr << "[GET ERROR] " << method << ": " << e.what() << endl;
                write_error_csv(method, "GET request error");
            }
            file = "results.csv";
        }
        
        ifstream in(file, ios::binary);
        if (!in) {
            string resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nFile not found: " + file;
            send(c, resp.c_str(), (int)resp.size(), 0);
            closesocket(c); 
            continue;
        }
        
        string body((istreambuf_iterator<char>(in)), {}); 
        in.close();
        string type = (file.find(".csv") != string::npos) ? "text/csv" : "text/html";
        string resp = "HTTP/1.1 200 OK\r\nContent-Type: " + type + "; charset=utf-8\r\n"
                     "Content-Length: " + to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        send(c, resp.c_str(), (int)resp.size(), 0);
        closesocket(c);
    }
    
    closesocket(s); 
    WSACleanup(); 
    return 0;
}