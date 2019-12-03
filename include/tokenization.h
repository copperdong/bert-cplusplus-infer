//
// Created by 刘佳伟 on 2019/6/19.
//

#ifndef FUPU_PROJECT_TOKENIZATION_H
#define FUPU_PROJECT_TOKENIZATION_H

#include <string>
#include <vector>

#include "base_type.h"


class WordpieceTokenizer{
private:
    std::wstring unk_token_;
    int max_input_chars_per_word_;
public:
    WordpieceTokenizer(std::wstring unk_token=L"[UNK]", int max_input_chars_per_word=100);
    void Tokenize(vocab_map &vm, std::wstring &text, std::vector<std::wstring> &result);
};

class BasicTokenizer{
private:
    bool do_lower_case_;
    std::wstring RunStripAccents(std::wstring text);
    void RunSplitOnPunc(std::wstring text, std::vector<std::wstring> &result);
    bool IsChineseChar(wchar_t cp);
    std::wstring TokenizeChineseChars(std::wstring text);
    std::wstring CleanText(std::wstring text);
public:
    BasicTokenizer(bool do_lower_case= true);
    void Tokenize(std::wstring text, std::vector<std::wstring> &result);
};

class FullTokenizer{
private:
    vocab_map vm_;
    inv_vocab_map ivm_;
    BasicTokenizer *bt_ = nullptr;
    WordpieceTokenizer *wt_ = nullptr;

public:
    FullTokenizer(std::string &vocab_file, bool do_lower_case=true);
    ~FullTokenizer();
    void Tokenize(std::wstring &text, std::vector<std::wstring> &result);
    void ConvertTokensToIds(std::vector<std::wstring> &tokens, std::vector<int> &ids);
    void ConvertIdsToTokens(std::vector<int> &ids, std::vector<std::wstring> &tokens);
};

#endif //FUPU_PROJECT_TOKENIZATION_H
