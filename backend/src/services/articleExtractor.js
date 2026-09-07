const axios = require('axios');
const cheerio = require('cheerio');
const config = require('../config/env');

const MAX_WORDS_PER_ARTICLE = 15000;
const MIN_EXTRACTED_CHARS = 100;

/**
 * Checks if a URL is safe to fetch (SSRF prevention & sanity check)
 */
const isSafeUrl = (rawUrl) => {
  try {
    if (!rawUrl || typeof rawUrl !== 'string') return false;
    const parsed = new URL(rawUrl);

    // Only allow HTTP/HTTPS
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return false;
    }

    let hostname = parsed.hostname.toLowerCase();
    // Strip brackets for IPv6
    if (hostname.startsWith('[') && hostname.endsWith(']')) {
      hostname = hostname.slice(1, -1);
    }

    // Block localhost, loopbacks, internal DNS suffixes, cloud metadata hostnames
    const blockedHostnames = [
      'localhost',
      '127.0.0.1',
      '0.0.0.0',
      '::1',
      '169.254.169.254',
      'metadata.google.internal',
      'metadata.goog',
      'instance-data'
    ];
    if (blockedHostnames.includes(hostname)) {
      return false;
    }

    if (
      hostname.endsWith('.local') ||
      hostname.endsWith('.internal') ||
      hostname.endsWith('.lan') ||
      hostname.endsWith('.corp') ||
      hostname.endsWith('.home')
    ) {
      return false;
    }

    // Check for IPv6 link-local (fe80::/10), unique local (fc00::/7, fd00::/8), or IPv4 mapped loopback
    if (
      hostname.startsWith('fe80:') ||
      hostname.startsWith('fc00:') ||
      hostname.startsWith('fd00:') ||
      hostname.startsWith('::ffff:127.')
    ) {
      return false;
    }

    // Block hexadecimal or octal IP representations (e.g., 0x7f.1, 0177.0.0.1)
    if (/^0x[0-9a-f]+$/i.test(hostname) || /^0[0-7]+(\.[0-7]+)*$/.test(hostname)) {
      return false;
    }

    // Block pure integer IP representation (e.g., 2130706433)
    if (/^\d+$/.test(hostname)) {
      return false;
    }

    // Block standard private and reserved IPv4 ranges
    const ipv4Regex = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;
    const ipMatch = hostname.match(ipv4Regex);
    if (ipMatch) {
      const oct1 = parseInt(ipMatch[1], 10);
      const oct2 = parseInt(ipMatch[2], 10);
      const oct3 = parseInt(ipMatch[3], 10);
      const oct4 = parseInt(ipMatch[4], 10);

      if (oct1 > 255 || oct2 > 255 || oct3 > 255 || oct4 > 255) return false;
      if (oct1 === 0) return false; // 0.0.0.0/8
      if (oct1 === 10) return false; // 10.0.0.0/8
      if (oct1 === 127) return false; // Loopback 127.0.0.0/8
      if (oct1 === 169 && oct2 === 254) return false; // Link-local / Cloud metadata 169.254.0.0/16
      if (oct1 === 172 && oct2 >= 16 && oct2 <= 31) return false; // 172.16.0.0/12
      if (oct1 === 192 && oct2 === 168) return false; // 192.168.0.0/16
      if (oct1 === 100 && oct2 >= 64 && oct2 <= 127) return false; // Shared address space 100.64.0.0/10
    }

    // Block non-HTML binary media extensions
    const path = parsed.pathname.toLowerCase();
    const blockedExts = ['.pdf', '.zip', '.exe', '.tar', '.gz', '.mp3', '.mp4', '.avi', '.jpg', '.jpeg', '.png', '.gif', '.iso', '.bin'];
    if (blockedExts.some(ext => path.endsWith(ext))) {
      return false;
    }

    return true;
  } catch (e) {
    return false;
  }
};

/**
 * Cleans extracted text, removing reference numbers, edit tags, and whitespace
 */
const cleanText = (raw) => {
  if (!raw) return '';
  return raw
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/\[\d+\]/g, '')
    .replace(/\[[a-z]\]/g, '')
    .replace(/\[edit\]/gi, '')
    .replace(/\[update\]/gi, '')
    .replace(/\s+/g, ' ')
    .trim();
};

/**
 * Caps text to a maximum number of words
 */
const capWords = (text, maxWords = MAX_WORDS_PER_ARTICLE) => {
  if (!text) return '';
  const words = text.split(/\s+/);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(' ');
};

/**
 * Extracts readable article body text from HTML page
 * Step 5: Web Content Extraction Hardening
 */
const extractArticleText = async (article, timeoutMs = config.extractionTimeoutMs) => {
  const url = article.url;

  if (!isSafeUrl(url)) {
    return {
      success: false,
      url,
      reason: 'URL rejected by security safety filter (SSRF / invalid protocol / media file)',
      text: article.description ? cleanText(article.description) : ''
    };
  }

  try {
    const response = await axios.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
      },
      timeout: timeoutMs || 8000,
      maxRedirects: 10,
      maxContentLength: 5 * 1024 * 1024, // 5MB limit
      validateStatus: (status) => status >= 200 && status < 400
    });

    const contentType = (response.headers['content-type'] || '').toLowerCase();
    if (contentType && !contentType.includes('text/html') && !contentType.includes('application/xhtml+xml') && !contentType.includes('text/plain')) {
      throw new Error(`Rejected non-text content-type: ${contentType}`);
    }

    const html = response.data;
    if (typeof html !== 'string') {
      throw new Error('Response is not HTML string');
    }

    const $ = cheerio.load(html);

    // Remove boilerplate elements common to all sites
    $(
      'script, style, noscript, nav, header, footer, svg, form, iframe, aside, ' +
      'select, button, .ad, .ads, .advertisement, .social-share, .cookie-banner, ' +
      '.menu, .navigation, .sidebar, .comments, #comments, .related-posts, .popup, ' +
      '.author-bio, .byline, audio, video'
    ).remove();

    const isWiki = url.toLowerCase().includes('wikipedia.org');

    if (isWiki) {
      // Wikipedia specific noise stripping
      $(
        'sup.reference, .mw-editsection, .hatnote, .infobox, .navbox, .navbox-styles, ' +
        '.metadata, .ambox, .catlinks, .toc, #toc, .vector-toc, .thumb, .thumbinner, ' +
        '.gallery, figure, figcaption, .reflist, #References, #See_also, #External_links'
      ).remove();
    }

    // Select primary content container
    const containers = isWiki ? [
      '.mw-parser-output',
      '#mw-content-text',
      'body'
    ] : [
      'article',
      'main',
      '[role="main"]',
      '.article-content',
      '.story-body',
      '.entry-content',
      '.post-content',
      'body'
    ];

    let root = null;
    let maxPCount = 0;
    for (const sel of containers) {
      $(sel).each((_, el) => {
        const pCount = $(el).find('p').length;
        if (pCount > maxPCount) {
          maxPCount = pCount;
          root = $(el);
        }
      });
      if (maxPCount >= 3) break;
    }
    if (!root) root = $('body');

    // Extract substantive paragraphs
    const paragraphs = [];
    root.find('p').each((_, el) => {
      const pText = cleanText($(el).text());
      // Discard micro-fragments, button labels, copyright lines
      if (pText.length >= 35 && !pText.toLowerCase().startsWith('copyright') && !pText.toLowerCase().startsWith('all rights reserved')) {
        paragraphs.push(pText);
      }
    });

    let combinedText = paragraphs.join('\n\n');

    // Fallback: If extracted body is < 100 characters, fall back to snippet/description
    if (combinedText.length < MIN_EXTRACTED_CHARS && article.description) {
      return {
        success: true,
        url,
        text: cleanText(article.description),
        usedFallback: true,
        paragraphCount: 1
      };
    }

    // Cap text to maximum words to prevent memory explosion
    combinedText = capWords(combinedText, MAX_WORDS_PER_ARTICLE);

    return {
      success: combinedText.length >= MIN_EXTRACTED_CHARS,
      url,
      text: combinedText,
      usedFallback: false,
      paragraphCount: paragraphs.length
    };
  } catch (error) {
    // Graceful fallback to article description on extraction failure
    const fallbackText = article.description ? cleanText(article.description) : '';
    return {
      success: fallbackText.length >= 40,
      url,
      reason: error.message,
      text: fallbackText,
      usedFallback: fallbackText.length > 0,
      paragraphCount: fallbackText ? 1 : 0
    };
  }
};

module.exports = {
  extractArticleText,
  isSafeUrl,
  cleanText,
  capWords
};
