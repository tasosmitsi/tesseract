#pragma once
#include <string>
#include <sstream>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

// ===============================
// AI generated, use with caution
// ===============================

namespace expr_diag
{

    inline int tensor_count = 0;
    inline int scalar_count = 0;

    // Greek letters for scalars
    inline std::string get_greek_letter(int n)
    {
        const char *greeks[] = {"α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"};
        constexpr int num_greeks = 23;
        if (n < num_greeks)
        {
            return greeks[n];
        }
        // Fallback for many scalars
        return greeks[n % num_greeks] + std::to_string(n / num_greeks);
    }

    // Unicode subscript digits
    inline std::string to_subscript(int n)
    {
        const char *subscripts[] = {"₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"};
        if (n == 0)
            return subscripts[0];

        std::string result;
        while (n > 0)
        {
            result = subscripts[n % 10] + result;
            n /= 10;
        }
        return result;
    }

    // Subscript type characters
    inline std::string to_subscript_type(const std::string &type_name)
    {
        if (type_name == "double")
            return "ᵈ";
        else if (type_name == "float")
            return "ᶠ";
        else if (type_name == "int")
            return "ⁱ";
        else if (type_name == "long")
            return "ˡ";
        else if (type_name == "unsigned")
            return "ᵘ";
        else
            return "?";
    }

    // Unicode superscript digits (for permutation indices)
    inline std::string to_superscript(int n)
    {
        const char *superscripts[] = {"⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"};
        if (n == 0)
            return superscripts[0];

        std::string result;
        while (n > 0)
        {
            result = superscripts[n % 10] + result;
            n /= 10;
        }
        return result;
    }

    inline void skip_ws(const std::string &s, size_t &pos)
    {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n'))
            pos++;
    }

    inline std::string read_ident(const std::string &s, size_t &pos)
    {
        size_t start = pos;
        while (pos < s.size() && (isalnum(s[pos]) || s[pos] == '_'))
            pos++;
        return s.substr(start, pos - start);
    }

    // Read a number (possibly with 'ul' suffix)
    inline std::string read_number(const std::string &s, size_t &pos)
    {
        std::string num;
        while (pos < s.size() && isdigit(s[pos]))
        {
            num += s[pos++];
        }
        // Skip 'ul', 'lu', 'u', 'l' suffixes
        while (pos < s.size() && (s[pos] == 'u' || s[pos] == 'l' || s[pos] == 'U' || s[pos] == 'L'))
        {
            pos++;
        }
        return num;
    }

    inline std::string parse_expr(const std::string &s, size_t &pos);

    inline std::string parse_tensor(const std::string &s, size_t &pos)
    {
        // We're right after "FusedTensorND"
        pos++; // skip '<'
        skip_ws(s, pos);

        // Read type
        std::string type_name = read_ident(s, pos);
        std::string type_char = (type_name == "double")  ? "d"
                                : (type_name == "float") ? "f"
                                : (type_name == "int")   ? "i"
                                : (type_name == "long")  ? "l"
                                                         : "?";

        // Read dimensions
        std::vector<std::string> dims;
        while (pos < s.size())
        {
            skip_ws(s, pos);
            if (s[pos] == '>')
            {
                pos++;
                break;
            }
            if (s[pos] == ',')
            {
                pos++;
                skip_ws(s, pos);
                std::string dim = read_number(s, pos);
                if (!dim.empty())
                {
                    dims.push_back(dim);
                }
            }
        }

        // Build: T₀d₃×₃
        std::string result = "T" + to_subscript(tensor_count++) + type_char;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (i > 0)
                result += "×";
            result += to_subscript(std::stoi(dims[i]));
        }

        return result;
    }

    inline std::string parse_permuted_view(const std::string &s, size_t &pos)
    {
        // We're right after "PermutedViewConstExpr"
        pos++; // skip '<'
        skip_ws(s, pos);

        // Parse inner expression
        std::string inner = parse_expr(s, pos);

        // Read permutation indices
        std::vector<int> perm;
        while (pos < s.size())
        {
            skip_ws(s, pos);
            if (s[pos] == '>')
            {
                pos++;
                break;
            }
            if (s[pos] == ',')
            {
                pos++;
                skip_ws(s, pos);
                std::string idx = read_number(s, pos);
                if (!idx.empty())
                {
                    perm.push_back(std::stoi(idx));
                }
            }
        }

        // Check if it's a simple transpose (swap of 2 dims: 1,0)
        if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0)
        {
            return inner + "ᵀ";
        }

        // General permutation: show as superscript indices
        std::string result = inner + "⁽";
        for (size_t i = 0; i < perm.size(); ++i)
        {
            if (i > 0)
                result += "'";
            result += to_superscript(perm[i]);
        }
        result += "⁾";

        return result;
    }

    inline std::string make_scalar(const std::string &scalar_type)
    {
        std::string type_char = (scalar_type == "double")  ? "d"
                                : (scalar_type == "float") ? "f"
                                : (scalar_type == "int")   ? "i"
                                : (scalar_type == "long")  ? "l"
                                                           : "?";
        return type_char + get_greek_letter(scalar_count++);
    }

    inline std::string parse_expr(const std::string &s, size_t &pos)
    {
        skip_ws(s, pos);
        std::string name = read_ident(s, pos);
        skip_ws(s, pos);

        if (name == "FusedTensorND")
        {
            return parse_tensor(s, pos);
        }
        else if (name == "PermutedViewConstExpr")
        {
            return parse_permuted_view(s, pos);
        }
        else if (name == "BinaryExpr")
        {
            pos++; // skip '<'
            std::string lhs = parse_expr(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            std::string rhs = parse_expr(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            skip_ws(s, pos);
            std::string op = read_ident(s, pos);
            skip_ws(s, pos);
            pos++; // skip '>'

            std::string sym = (op == "Add")   ? " + "
                              : (op == "Sub") ? " − "
                              : (op == "Mul") ? " · "
                              : (op == "Div") ? " / "
                              : (op == "Min") ? " ∧ "
                              : (op == "Max") ? " ∨ "
                                              : " ? ";
            return "(" + lhs + sym + rhs + ")";
        }
        else if (name == "ScalarExprRHS")
        {
            pos++; // skip '<'
            std::string expr = parse_expr(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            skip_ws(s, pos);
            std::string scalar_type = read_ident(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            skip_ws(s, pos);
            std::string op = read_ident(s, pos);
            skip_ws(s, pos);
            pos++; // skip '>'

            std::string scalar = make_scalar(scalar_type);
            std::string sym = (op == "Add")   ? " + "
                              : (op == "Sub") ? " − "
                              : (op == "Mul") ? " · "
                              : (op == "Div") ? " / "
                              : (op == "Min") ? " ∧ "
                              : (op == "Max") ? " ∨ "
                                              : " ? ";
            return "(" + expr + sym + scalar + ")";
        }
        else if (name == "ScalarExprLHS")
        {
            pos++; // skip '<'
            std::string expr = parse_expr(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            skip_ws(s, pos);
            std::string scalar_type = read_ident(s, pos);
            skip_ws(s, pos);
            pos++; // skip ','
            skip_ws(s, pos);
            std::string op = read_ident(s, pos);
            skip_ws(s, pos);
            pos++; // skip '>'

            // Negation special case: 0 - expr
            if (op == "Sub")
            {
                return "−" + expr;
            }

            std::string scalar = make_scalar(scalar_type);
            std::string sym = (op == "Add")   ? " + "
                              : (op == "Sub") ? " − "
                              : (op == "Mul") ? " · "
                              : (op == "Div") ? " / "
                                              : " ? ";
            return "(" + scalar + sym + expr + ")";
        }

        return "?";
    }

    template <typename Expr>
    std::string to_string()
    {
        tensor_count = 0;
        scalar_count = 0;
        std::string type_name = typeid(Expr).name();

#if defined(__GNUC__) || defined(__clang__)
        int status;
        char *demangled = abi::__cxa_demangle(type_name.c_str(), nullptr, nullptr, &status);
        if (status == 0)
        {
            type_name = demangled;
            free(demangled);
        }
#endif

        size_t pos = 0;
        return parse_expr(type_name, pos);
    }

    template <typename Expr>
    void print_expr()
    {
        std::cout << to_string<Expr>() << "\n";
    }

    inline void print_legend()
    {
        std::cout << "═══════════════════════════════════════════════\n";
        std::cout << "  Expression Notation Legend\n";
        std::cout << "═══════════════════════════════════════════════\n";
        std::cout << "  Tensors:  Tₙtᵢ×ⱼ = Tensor #n, type t, dims i×j\n";
        std::cout << "  Types:    d=double, f=float, i=int, l=long\n";
        std::cout << "  Scalars:  tα, tβ, tγ... (type prefix + greek)\n";
        std::cout << "            dα=double, fβ=float, iγ=int, lδ=long\n";
        std::cout << "  Views:    ᵀ = transpose (permutation 1,0)\n";
        std::cout << "            ⁽ⁱ'ʲ'ᵏ⁾ = general permutation\n";
        std::cout << "  Ops:      + − · /\n";
        std::cout << "═══════════════════════════════════════════════\n";
    }

} // namespace expr_diag
