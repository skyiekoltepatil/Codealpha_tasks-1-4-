document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const sourceLang = document.getElementById('sourceLang');
    const targetLang = document.getElementById('targetLang');
    const swapBtn = document.getElementById('swapBtn');
    const translateBtn = document.getElementById('translateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const copyBtn = document.getElementById('copyBtn');
    const charCount = document.getElementById('charCount');
    
    const MAX_CHARS = 500;

    // Update Character Count
    inputText.addEventListener('input', () => {
        let text = inputText.value;
        if (text.length > MAX_CHARS) {
            inputText.value = text.substring(0, MAX_CHARS);
        }
        charCount.textContent = `${inputText.value.length}/${MAX_CHARS}`;
    });

    // Clear Text
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        outputText.value = '';
        charCount.textContent = `0/${MAX_CHARS}`;
        inputText.focus();
    });

    // Swap Languages
    swapBtn.addEventListener('click', () => {
        const tempLang = sourceLang.value;
        sourceLang.value = targetLang.value;
        targetLang.value = tempLang;
        
        const tempText = inputText.value;
        inputText.value = outputText.value;
        outputText.value = tempText;
        
        charCount.textContent = `${inputText.value.length}/${MAX_CHARS}`;
    });

    // Copy to Clipboard
    const toast = document.getElementById('toast');
    const showToast = (message = 'Text copied to clipboard!') => {
        if (!toast) return;
        toast.textContent = message;
        toast.classList.remove('hide');
        setTimeout(() => {
            toast.classList.add('hide');
        }, 2200);
    };

    copyBtn.addEventListener('click', () => {
        if (!outputText.value) return;
        navigator.clipboard.writeText(outputText.value).then(() => {
            const icon = copyBtn.querySelector('i');
            icon.className = 'fa-solid fa-check';
            icon.style.color = '#10B981';
            showToast();
            
            setTimeout(() => {
                icon.className = 'fa-regular fa-copy';
                icon.style.color = '';
            }, 2000);
        });
    });

    // Translate Function
    const translateText = async () => {
        const text = inputText.value.trim();
        if (!text) return;

        const src = sourceLang.value;
        const tgt = targetLang.value;

        // UI State Update
        const btnText = translateBtn.querySelector('.btn-text');
        const btnIcon = translateBtn.querySelector('.fa-arrow-right');
        const loader = translateBtn.querySelector('.loader');

        btnText.textContent = 'Translating';
        btnIcon.classList.add('hide');
        loader.classList.remove('hide');
        translateBtn.disabled = true;
        translateBtn.style.opacity = '0.7';

        try {
            const response = await fetch(`https://translate.googleapis.com/translate_a/single?client=gtx&sl=${src}&tl=${tgt}&dt=t&q=${encodeURIComponent(text)}`);
            const data = await response.json();
            
            if (data && data[0]) {
                const translated = data[0].map(item => item[0]).join('');
                outputText.value = translated;
            } else {
                outputText.value = "Translation error.";
            }
        } catch (error) {
            console.error('Translation error:', error);
            outputText.value = "Network error. Please try again.";
        } finally {
            // Reset UI State
            btnText.textContent = 'Translate';
            btnIcon.classList.remove('hide');
            loader.classList.add('hide');
            translateBtn.disabled = false;
            translateBtn.style.opacity = '1';
        }
    };

    translateBtn.addEventListener('click', translateText);
    
    // Allow translation with Ctrl+Enter / Cmd+Enter
    inputText.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            translateText();
        }
    });
});
