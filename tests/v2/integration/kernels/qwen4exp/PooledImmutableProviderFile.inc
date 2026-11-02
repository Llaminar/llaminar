/** @file PooledImmutableProviderFile.inc
 * @brief Original sparse/canonical provider bytes shared by setup qualification and captured lookup fixtures.
 * The file format, row authoring and original decoder oracle are unchanged.
 */
/** @brief Fail the exact requested native cell without a skip or alternate path. */
void requireProvider(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
/** @brief Author original rows at their full source coordinates; sparse holes are file layout, never transmitted padding. */
class ProviderFile final {
public:
    static constexpr std::uint64_t High=0x100000007ULL,Rows=0x100000100ULL;
    static constexpr std::size_t Columns=160,RowBytes=90,Offset=37;
    /** @brief Keep the original sparse IQ4_NL high-row diagnostic geometry. */
    ProviderFile(){
        open();
        try{
            requireProvider(::ftruncate(fd,static_cast<off_t>(Offset+rows*row_bytes))==0,"Cannot declare exact sparse provider extent");
            for(auto row:std::array<std::uint64_t,7>{7,101,102,700,701,900,High})author(row,legacy(row));
            for(std::uint64_t row=2000;row<2016;++row)author(row,legacy(row));
        }catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief Author canonical original-native fixture tensor bytes, never a conversion in the production path.
     * @param native Canonical test inventory's original native tensor. @param sourceType Actual loader GGUF discriminator.
     * @param width Original complete native row width. */
    ProviderFile(const TensorBase& native,GGUFTensorType sourceType,std::size_t width)
        :rows(97),columns(width),format(sourceType),type(native.native_type()){
        row_bytes=descriptor().rowBytes();requireProvider(native.size_bytes()==rows*row_bytes,"Canonical source row geometry differs from original native tensor");open();
        try{
            requireProvider(::ftruncate(fd,static_cast<off_t>(Offset+rows*row_bytes))==0,"Cannot declare canonical provider extent");
            const auto* bytes=static_cast<const std::byte*>(native.raw_data());
            for(std::uint64_t row=0;row<rows;++row)author(row,{bytes+row*row_bytes,bytes+(row+1)*row_bytes});
        }catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /**
     * @brief Author the same original IQ4_NL bytes for a fixed performance trajectory.
     * @param sourceRows Original reduced component table extent, not a full-model claim.
     * @param selected Original rows consumed by every untimed/timed trace phase.
     */
    ProviderFile(std::uint64_t sourceRows,std::span<const std::uint64_t> selected):rows(sourceRows){
        requireProvider(rows && rows<=static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()-Offset)/row_bytes,"Replay source file extent overflows off_t");
        open();try{requireProvider(::ftruncate(fd,static_cast<off_t>(Offset+rows*row_bytes))==0,"Cannot declare replay provider extent");
            for(const auto row:selected)if(!authored.contains(row)){std::vector<std::byte> bytes(RowBytes);ngram_replay::fillRow(row,bytes);author(row,std::move(bytes));}
        }catch(...){::close(fd);::unlink(path.c_str());throw;}
    }
    /** @brief The actual source owner is gone before its original file unlinks. */
    ~ProviderFile(){::close(fd);::unlink(path.c_str());}
    /** @brief Require an independently authored byte oracle before native input publication; sparse holes are not invented reference rows. */
    void requireAuthored(std::span<const std::uint64_t> selected)const{
        for(const auto row:selected)if(!authored.contains(row))
            throw std::invalid_argument("Pooled replay requested an unauthored source row: "+std::to_string(row));
    }
    /** @return Exact file descriptor consumed by the existing production provider. */
    ImmutableRowTableSource descriptor()const{return {.path=path,.tensor_name="per_layer_token_embd.weight",.file_offset=Offset,.rows=rows,.row_elements=columns,.format=format};}
    /** @return Authored immutable source bytes for the terminal independent CPU decoder oracle. */
    std::span<const std::byte> original(std::uint64_t row)const{return authored.at(row);}
    std::uint64_t rows=Rows;std::size_t columns=Columns,row_bytes=RowBytes;GGUFTensorType format=GGUFTensorType::IQ4_NL;TensorType type=TensorType::IQ4_NL;
    int fd=-1;std::string path;
private:
    /** @brief Create only fixture source I/O, never an alternate delivery publisher. */
    void open(){auto name=(std::filesystem::temp_directory_path()/"qfn-real-provider-XXXXXX").string();
        fd=::mkstemp(name.data());requireProvider(fd>=0,"Cannot create original provider file");path=std::move(name);}
    /** @brief Write one complete original native row at its real declared source coordinate. */
    void author(std::uint64_t row,std::vector<std::byte> bytes){
        requireProvider(row<rows&&bytes.size()==row_bytes,"Invalid authored original source row");
        requireProvider(::pwrite(fd,bytes.data(),bytes.size(),static_cast<off_t>(Offset+row*row_bytes))==static_cast<ssize_t>(bytes.size()),"Cannot author original provider bytes");authored.emplace(row,std::move(bytes));
    }
    /** @return Original sparse IQ4_NL source pattern, preserving the prior high-row test oracle. */
    static std::vector<std::byte> legacy(std::uint64_t row){
        std::vector<std::byte> result(RowBytes);
        const auto signature=row^(row>>32); // Preserve the original functional fixture's authored bytes.
        for(unsigned block=0;block<5;++block){result[block*18]=std::byte{0};result[block*18+1]=std::byte{0x3c};
            for(unsigned code=0;code<16;++code)result[block*18+2+code]=std::byte((signature+block+code)%16|(((signature*3+block+code)%16)<<4));}
        return result;
    }
    std::map<std::uint64_t,std::vector<std::byte>> authored;
};
