//
// Created by 刘佳伟 on 2019/6/19.
//

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <codecvt>
#include "unicode.h"
#include "uninorms.h"

#include "base_type.h"
#include "tokenization.h"

using namespace ufal::unilib;

std::u32string to_utf32(std::string str) {
    return std::wstring_convert < std::codecvt_utf8 < char32_t > , char32_t > ().from_bytes(str);
    //std::u32string str32=to_utf<char32_t>(str, "UTF-8");
    //return str32;
}

std::string to_utf8(std::u32string str32) {
    return std::wstring_convert < std::codecvt_utf8_utf16 < char32_t > , char32_t > ().to_bytes(str32);
    //std::string str=from_utf(str32, "UTF-32");
    //return str;
}

std::wstring to_wchar_t(std::string str) {
    return std::wstring_convert < std::codecvt_utf8 < wchar_t > , wchar_t > ().from_bytes(str);
    //return to_utf<wchar_t>(str, "UTF-8");
}

std::string to_utf8(std::wstring wstr) {
    return std::wstring_convert < std::codecvt_utf8_utf16 < wchar_t > , wchar_t > ().to_bytes(wstr);
    //return from_utf(wstr, "UTF-8");
}

void load_vocab(std::string vocab_file, vocab_map &vm) {
    std::ifstream in(vocab_file);
    std::string line;
    if (in) {
        int index = 0;
        while (std::getline(in, line)) {
            vm.insert(std::make_pair(boost::trim_copy(to_wchar_t(line)), index));
            index++;
        }
    }
}


void inverse_vocab_map(vocab_map &vm, inv_vocab_map &ivm) {
    std::unordered_map<std::wstring, int>::iterator it = vm.begin();
    while (it != vm.end()) {
        ivm.insert(std::make_pair(it->second, it->first));
        ++it;
    }
}


void whitespace_tokenize(std::wstring text, std::vector <std::wstring> &tokens_vec) {
    std::wstring trim_text = boost::trim_copy(text);
    boost::split(tokens_vec, trim_text, boost::is_any_of(" \t\n\r"), boost::token_compress_on);
}


bool is_whitespace(char32_t tmp_char) {
    if (tmp_char == ' ' || tmp_char == '\t' || tmp_char == '\n' || tmp_char == '\r')
        return true;
    unicode::category_t cat = unicode::category(tmp_char);
    if (cat == unicode::Zs)
        return true;
    return false;
}

//todo(jiawei): 可以通过与操作简单的判断unicode的种类，不用通过
bool is_punctuation(char32_t tmp_char) {
    if (tmp_char >= 33 && tmp_char <= 47 || tmp_char >= 58 && tmp_char <= 64 || tmp_char >= 91 && tmp_char <= 96 ||
        tmp_char >= 123 && tmp_char <= 126)
        return true;
    unicode::category_t cat = unicode::category(tmp_char);
    if (cat == unicode::Pc || cat == unicode::Pd || cat == unicode::Pe || cat == unicode::Pf || cat == unicode::Pi ||
        cat == unicode::Po || cat == unicode::Ps)
        return true;
    return false;
}


bool is_control(char32_t tmp_char) {
    if (tmp_char == '\t' || tmp_char == '\n' || tmp_char == '\r')
        return false;
    unicode::category_t cat = unicode::category(tmp_char);
    if (cat == unicode::Cc || cat == unicode::Cf || cat == unicode::Cn || cat == unicode::Co || cat == unicode::Cs)
        return true;
    return false;
}


WordpieceTokenizer::WordpieceTokenizer(std::wstring unk_token, int max_input_chars_per_word) {
    unk_token_ = unk_token;
    max_input_chars_per_word_ = max_input_chars_per_word;
}


//TODO(jiawei): 看std::string的length方法返回的值会不会包含了/0.
void WordpieceTokenizer::Tokenize(vocab_map &vm, std::wstring &text, std::vector <std::wstring> &result) {
    std::vector <std::wstring> tokens_vec;
    whitespace_tokenize(text, tokens_vec);
    int token_num = tokens_vec.size();

    for (int i = 0; i < token_num; i++) {
        int token_len = tokens_vec[i].length();
        if (token_len > max_input_chars_per_word_) {
            result.push_back(unk_token_);
            continue;
        }

        bool is_bad = false;
        int start_index = 0;
        std::vector <std::wstring> sub_tokens;

        while (start_index < token_len) {
            int end_index = token_len;
            std::wstring cur_substr;

            while (start_index < end_index) {
                std::wstring substr = tokens_vec[i].substr(start_index, end_index - start_index);
                if (start_index > 0) {
                    substr = L"##" + substr;
                }
                if (vm.find(substr) != vm.end()) {
                    cur_substr = substr;
                    break;
                }
                end_index -= 1;
            }
            if (cur_substr.empty()) {
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start_index = end_index;
        }

        if (is_bad) {
            result.push_back(unk_token_);
        } else {
            result.insert(result.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
}


BasicTokenizer::BasicTokenizer(bool do_lower_case) {
    do_lower_case_ = do_lower_case;
}

std::wstring BasicTokenizer::RunStripAccents(std::wstring text) {
    std::u32string text32 = to_utf32(to_utf8(text));
    uninorms::nfd(text32);

    int len = text32.length();
    const char32_t *p = text32.c_str();
    char32_t *q = new char32_t[len+1];
    int j = 0;
    for (int i = 0; i < len; i++) {
        unicode::category_t cat = unicode::category(p[i]);
        if (cat == unicode::Mn)
            continue;
        q[j++] = p[i];
    }
    q[j] = '\0';
    std::u32string result32;
    result32 = q;

    //std::cout << to_utf8(result32) << std::endl;

    delete[]q;
    return to_wchar_t(to_utf8(result32));
}

void BasicTokenizer::RunSplitOnPunc(std::wstring text, std::vector <std::wstring> &result) {
    std::u32string text32 = to_utf32(to_utf8(text));
    const char32_t *p = text32.c_str();
    int len = text32.length();
    std::u32string word32;
    int index = 0;
    char32_t *total_word = new char32_t[100]; //假定最大的单词的长度为100个字符
    for (int i = 0; i < len; i++) {
        char32_t temp_char = p[i];
        if (is_punctuation(temp_char)) {
            if (index > 0) {
                total_word[index] = '\0';
                word32 = total_word;
                result.push_back(to_wchar_t(to_utf8(word32)));
                index = 0;
                delete[]total_word;
                total_word = new char32_t[100];
            }

            char32_t *q = new char32_t[2];
            q[0] = temp_char;
            q[1] = '\0';
            word32 = q;
            delete []q;
            result.push_back(to_wchar_t(to_utf8(word32)));
        } else {
            total_word[index++] = temp_char;
        }
    }
    if (index > 0) {
        total_word[index] = '\0';
        word32 = total_word;
        result.push_back(to_wchar_t(to_utf8(word32)));
    }

    delete[]total_word;
}

bool BasicTokenizer::IsChineseChar(wchar_t zh) {
    bool zh_flag = (zh >= 0x4E00 && zh <= 0x9FFF || zh >= 0x3400 && zh <= 0x4DBF ||
                    zh >= 0x20000 && zh <= 0x2A6DF ||
                    (zh >= 0x2A700 && zh <= 0x2B73F) ||
                    (zh >= 0x2B740 && zh <= 0x2B81F) ||
                    (zh >= 0x2B820 && zh <= 0x2CEAF) ||
                    (zh >= 0xF900 && zh <= 0xFAFF) ||
                    (zh >= 0x2F800 && zh <= 0x2FA1F));
    if (zh_flag)
        return true;
    return false;
}

std::wstring BasicTokenizer::TokenizeChineseChars(std::wstring text) {
    int wchar_num = text.length();
    const wchar_t *p = text.c_str();
    std::wstring result = L"";
    for (int i = 0; i < wchar_num; i++) {
        std::wstring temp;
        wchar_t p_temp[1];
        p_temp[0] = p[i];
        temp = p_temp;
        if (IsChineseChar(p[i])) {
            result += L" ";
            result += temp;
            result += L" ";
        } else {
            result += temp;
        }
    }

    return result;
}

std::wstring BasicTokenizer::CleanText(std::wstring text) {
    int wchar_num = text.length();
    const wchar_t *p = text.c_str();
    std::wstring result = L"";
    for (int i = 0; i < wchar_num; i++) {
        if (p[i] == 0 || p[i] == 0xfffd || is_control((char32_t) p[i]))
            continue;
        if (is_whitespace((char32_t) p[i])) {
            result += L" ";
        } else {
            std::wstring temp;
            wchar_t p_temp[1];
            p_temp[0] = p[i];
            temp = p_temp;
            result += temp;
        }
    }
    return result;
}

void BasicTokenizer::Tokenize(std::wstring text, std::vector <std::wstring> &result) {
    std::wstring clean_text = CleanText(text);
    std::wstring tcc = TokenizeChineseChars(clean_text);
    std::vector <std::wstring> tokens_vec;
    whitespace_tokenize(tcc, tokens_vec);
    int token_size = tokens_vec.size();

    /**
    for (int i = 0; i < token_size; i++) {
        std::wstring temp_token = tokens_vec[i];

        if (do_lower_case_) {
            std::transform(temp_token.begin(), temp_token.end(), temp_token.begin(), ::tolower);
            temp_token = run_strip_accents_(temp_token);
        }

        std::cout << to_utf8(temp_token) << std::endl;

    }

    exit(0);**/

    for (int i = 0; i < token_size; i++) {
        std::wstring temp_token = tokens_vec[i];

        if (do_lower_case_) {
            std::transform(temp_token.begin(), temp_token.end(), temp_token.begin(), ::tolower);
            temp_token = RunStripAccents(temp_token);
        }

        std::vector <std::wstring> temp_token_tokens;
        RunSplitOnPunc(temp_token, temp_token_tokens);
        result.insert(result.end(), temp_token_tokens.begin(), temp_token_tokens.end());
    }
}

FullTokenizer::FullTokenizer(std::string &vocab_file, bool do_lower_case) {
    load_vocab(vocab_file, vm_);
    inverse_vocab_map(vm_, ivm_);
    bt_ = new BasicTokenizer(do_lower_case);
    wt_ = new WordpieceTokenizer();
}

FullTokenizer::~FullTokenizer() {
    delete bt_;
    delete wt_;
}

void FullTokenizer::Tokenize(std::wstring &text, std::vector <std::wstring> &result) {
    std::vector <std::wstring> basic_tokens;
    bt_->Tokenize(text, basic_tokens);
    int basic_tokens_num = basic_tokens.size();
    for (int i = 0; i < basic_tokens_num; i++) {
        std::vector <std::wstring> wordpiece_token;
        wt_->Tokenize(vm_, basic_tokens[i], wordpiece_token);
        result.insert(result.end(), wordpiece_token.begin(), wordpiece_token.end());
    }

    /*
    int tokens_num = result.size();
    std::cout << tokens_num << std::endl;
    for (int i = 0; i < tokens_num; i++) {
        std::cout << to_utf8(result[i]) << std::endl;
    }
*/
}

void FullTokenizer::ConvertTokensToIds(std::vector <std::wstring> &tokens, std::vector<int> &ids) {
    int token_nums = tokens.size();
    for (int i = 0; i < token_nums; i++) {
        ids.push_back(vm_[tokens[i]]);
    }
}

void FullTokenizer::ConvertIdsToTokens(std::vector<int> &ids, std::vector <std::wstring> &tokens) {
    int id_nums = ids.size();
    for (int i = 0; i < id_nums; i++) {
        tokens.push_back(ivm_[i]);
    }
}

/*
int main() {
    std::wstring test_case = L"I come to 上海去  上学 **7fdade 23.";
    std::string vocab_file = "/data/liujiawei/bert-model/uncased_L-12_H-768_A-12/vocab.txt";
    FullTokenizer ft(vocab_file);
    std::vector <std::wstring> result;
    ft.Tokenize(test_case, result);
    int tokens_num = result.size();
    std::cout << tokens_num << std::endl;
    for (int i = 0; i < tokens_num; i++) {
        std::cout << to_utf8(result[i]) << std::endl;
    }
    return 0;
}
*/




