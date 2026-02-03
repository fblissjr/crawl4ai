#!/usr/bin/env python3
"""
c4md - Crawl4AI Markdown Extractor CLI

A CLI tool for web crawling and content extraction using crawl4ai.
Supports multiple output modes: text, markdown, screenshot, pdf, metadata, and archive.

Usage:
    c4md https://example.com --mode text
    c4md https://example.com --mode screenshot -v
    c4md https://example.com --mode archive --output ./results
"""

import asyncio
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
    BFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

console = Console()


def url_to_slug(url: str) -> str:
    """Generate a filesystem-safe filename from a URL."""
    parsed = urlparse(url)
    # Combine domain and path, removing protocol
    slug = parsed.netloc + parsed.path
    # Replace non-alphanumeric chars with underscores
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    # Remove leading/trailing underscores and collapse multiples
    slug = re.sub(r"_+", "_", slug).strip("_")
    # Limit length
    return slug[:100] if slug else "page"


async def crawl_url(
    url: str,
    mode: str,
    headless: bool = True,
    timeout: int = 30,
    fit: bool = False,
    verbose: bool = False,
    deep: bool = False,
    max_pages: int = 10,
    no_images: bool = False,
    image_quality: int = 2,
    cache_mode_str: str = "bypass",
    url_pattern: Optional[str] = None,
):
    """
    Core async crawl function.

    Demonstrates AsyncWebCrawler context manager usage with
    BrowserConfig and CrawlerRunConfig for various output modes.

    When deep=True, uses BFSDeepCrawlStrategy to crawl the seed page
    plus one level of linked pages (limited by max_pages).

    Args:
        no_images: If True, strips images from markdown output (LLM-friendly)
        image_quality: Image filtering threshold 1-6 (higher = stricter filtering)
    """
    browser_config = BrowserConfig(
        headless=headless,
        viewport_width=1920,
        viewport_height=1080,
    )

    # Configure markdown generator
    # - fit: applies content pruning to remove boilerplate
    # - no_images: strips images from markdown (reduces tokens for LLMs)
    markdown_options = {}
    if no_images:
        markdown_options["ignore_images"] = True

    content_filter = None
    if fit:
        content_filter = PruningContentFilter(
            threshold=0.48,
            threshold_type="fixed",
            min_word_threshold=0,
        )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
        options=markdown_options if markdown_options else None,
    )

    # Set flags based on mode
    needs_screenshot = mode in ("screenshot", "archive")
    needs_pdf = mode in ("pdf", "archive")
    needs_mhtml = mode == "archive"

    # Configure deep crawl strategy if requested
    deep_crawl_strategy = None
    if deep:
        # Build filter chain if URL pattern specified
        filter_chain = None
        if url_pattern:
            filter_chain = FilterChain([
                URLPatternFilter(patterns=[url_pattern])
            ])

        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=1,  # Seed page + 1 level of links
            include_external=False,  # Stay on same domain
            max_pages=max_pages,
            filter_chain=filter_chain,
        )

    # Map CLI cache mode string to CacheMode enum
    cache_mode_map = {
        "bypass": CacheMode.BYPASS,
        "enabled": CacheMode.ENABLED,
        "read-only": CacheMode.READ_ONLY,
        "write-only": CacheMode.WRITE_ONLY,
    }
    cache_mode = cache_mode_map.get(cache_mode_str, CacheMode.BYPASS)

    crawler_config = CrawlerRunConfig(
        cache_mode=cache_mode,
        page_timeout=timeout * 1000,  # Convert to milliseconds
        screenshot=needs_screenshot,
        pdf=needs_pdf,
        capture_mhtml=needs_mhtml,
        markdown_generator=markdown_generator,
        deep_crawl_strategy=deep_crawl_strategy,
        image_score_threshold=image_quality,  # Filter low-quality images (1-6)
        verbose=verbose,
    )

    if verbose:
        console.print(f"[dim]Crawling {url}{'(deep)' if deep else ''}...[/dim]")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(url=url, config=crawler_config)

    # Deep crawl returns a list, single crawl returns one result
    if deep:
        # Filter out failed results
        successful = [r for r in results if r.success]
        if not successful:
            raise click.ClickException("Deep crawl failed - no pages succeeded")
        return successful
    else:
        if not results.success:
            raise click.ClickException(f"Crawl failed: {results.error_message}")
        return results


def extract_text_markdown(result, fit: bool = False) -> str:
    """
    Extract clean text markdown (no images/links clutter).

    Uses result.markdown.fit_markdown when --fit is enabled,
    otherwise uses result.markdown.raw_markdown.
    """
    if result.markdown is None:
        return ""

    if fit and result.markdown.fit_markdown:
        return result.markdown.fit_markdown
    return result.markdown.raw_markdown


def extract_full_markdown(result) -> tuple[str, str]:
    """
    Extract full markdown with citations and references.

    Returns:
        tuple: (markdown_with_citations, references_markdown)
    """
    if result.markdown is None:
        return "", ""

    content = result.markdown.markdown_with_citations or result.markdown.raw_markdown
    references = result.markdown.references_markdown or ""
    return content, references


def download_and_compress_images(
    result,
    output_dir: Path,
    quality: int = 80,
    max_width: int = 800,
) -> dict[str, Path]:
    """
    Download images from result.media and compress them locally.

    Args:
        result: CrawlResult with media dict containing images
        output_dir: Directory to save images
        quality: JPEG quality 1-100
        max_width: Resize images wider than this

    Returns:
        dict mapping original URLs to local file paths
    """
    import hashlib
    import httpx

    images = result.media.get("images", [])
    if not images:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    url_to_path = {}

    with httpx.Client(timeout=10, follow_redirects=True) as client:
        for img in images:
            src = img.get("src", "")
            if not src or not src.startswith("http"):
                continue

            try:
                # Generate filename from URL hash
                url_hash = hashlib.md5(src.encode()).hexdigest()[:12]
                local_path = output_dir / f"{url_hash}.jpg"

                # Skip if already downloaded
                if local_path.exists():
                    url_to_path[src] = local_path
                    continue

                # Download
                resp = client.get(src)
                resp.raise_for_status()

                # Compress
                img_data = Image.open(BytesIO(resp.content))
                if img_data.mode in ("RGBA", "P"):
                    img_data = img_data.convert("RGB")

                # Resize if needed
                if max_width and img_data.width > max_width:
                    ratio = max_width / img_data.width
                    new_height = int(img_data.height * ratio)
                    img_data = img_data.resize(
                        (max_width, new_height), Image.Resampling.LANCZOS
                    )

                img_data.save(local_path, "JPEG", quality=quality, optimize=True)
                url_to_path[src] = local_path

            except Exception:
                # Skip failed downloads silently
                continue

    return url_to_path


def download_images_as_base64(
    result,
    quality: int = 80,
    max_width: int = 800,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Download images and return as base64 data URIs.

    Args:
        result: CrawlResult with media dict containing images
        quality: JPEG quality 1-100
        max_width: Resize images wider than this
        verbose: Print progress

    Returns:
        dict mapping original URLs to base64 data URIs
    """
    import httpx

    images = result.media.get("images", [])
    if not images:
        return {}

    url_to_base64 = {}

    with httpx.Client(timeout=10, follow_redirects=True) as client:
        for img in images:
            src = img.get("src", "")
            if not src or not src.startswith("http"):
                continue

            try:
                # Download
                resp = client.get(src)
                resp.raise_for_status()

                # Compress
                img_data = Image.open(BytesIO(resp.content))
                if img_data.mode in ("RGBA", "P"):
                    img_data = img_data.convert("RGB")

                # Resize if needed
                if max_width and img_data.width > max_width:
                    ratio = max_width / img_data.width
                    new_height = int(img_data.height * ratio)
                    img_data = img_data.resize(
                        (max_width, new_height), Image.Resampling.LANCZOS
                    )

                # Convert to base64
                buffer = BytesIO()
                img_data.save(buffer, "JPEG", quality=quality, optimize=True)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                url_to_base64[src] = f"data:image/jpeg;base64,{b64}"

            except Exception:
                # Skip failed downloads silently
                continue

    return url_to_base64


def embed_images_in_markdown(markdown: str, url_to_base64: dict[str, str]) -> str:
    """
    Replace image URLs in markdown with base64 data URIs.

    Args:
        markdown: Markdown content with image URLs
        url_to_base64: Mapping of URLs to base64 data URIs

    Returns:
        Markdown with embedded images
    """
    result = markdown
    for url, data_uri in url_to_base64.items():
        # Replace both markdown image syntax and raw URLs
        result = result.replace(url, data_uri)
    return result


def extract_date_from_result(result) -> Optional[str]:
    """
    Extract a date from a crawl result using multiple strategies.

    Tries in order:
    1. Standard metadata fields (date, published, datePublished)
    2. Open Graph article:published_time
    3. Date patterns in the first 500 chars of markdown

    Returns:
        ISO date string (YYYY-MM-DD) or None if not found
    """
    from datetime import datetime
    import re

    # Strategy 1: Check metadata for common date fields
    if result.metadata:
        for key in ["date", "published", "datePublished", "article:published_time",
                    "og:article:published_time", "pubdate", "publish_date"]:
            if key in result.metadata and result.metadata[key]:
                date_str = result.metadata[key]
                # Try to parse ISO format
                try:
                    if "T" in date_str:
                        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        return dt.strftime("%Y-%m-%d")
                    return date_str[:10]  # Assume YYYY-MM-DD prefix
                except (ValueError, TypeError):
                    pass

    # Strategy 2: Look for date patterns in markdown content
    if result.markdown:
        # Get first 1000 chars where dates usually appear
        text = str(result.markdown)[:1000]

        # Common date patterns
        patterns = [
            # ISO: 2024-01-15
            (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
            # US: January 15, 2024 or Jan 15, 2024
            (r"((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})", None),
            # UK: 15 January 2024
            (r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})", None),
        ]

        for pattern, fmt in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if fmt:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                else:
                    # Try dateutil for flexible parsing
                    try:
                        from dateutil import parser
                        dt = parser.parse(date_str, fuzzy=True)
                        return dt.strftime("%Y-%m-%d")
                    except (ImportError, ValueError):
                        # Fallback: manual parsing for common formats
                        try:
                            # Try "Month DD, YYYY" format
                            dt = datetime.strptime(
                                date_str.replace(",", ""),
                                "%B %d %Y"
                            )
                            return dt.strftime("%Y-%m-%d")
                        except ValueError:
                            try:
                                # Try abbreviated month
                                dt = datetime.strptime(
                                    date_str.replace(",", "").replace(".", ""),
                                    "%b %d %Y"
                                )
                                return dt.strftime("%Y-%m-%d")
                            except ValueError:
                                pass

    return None


def sort_results_by_date(results: list, descending: bool = True) -> list:
    """
    Sort crawl results by extracted date.

    Results without dates are placed at the end.
    """
    def get_sort_key(result):
        date = extract_date_from_result(result)
        # Results with no date go to the end
        if date is None:
            return ("9999-99-99" if descending else "0000-00-00", result.url)
        return (date, result.url)

    return sorted(results, key=get_sort_key, reverse=descending)


def extract_metadata(result, url: str) -> dict:
    """
    Extract article metadata as a dictionary.

    TODO: Implement this function based on your use case.

    Consider what metadata matters most for your workflow:
    - Basic info: title, description, author, publish date
    - Content stats: word count, reading time
    - Link analysis: internal/external link counts, domains linked
    - Media inventory: image count, video count
    - SEO data: meta keywords, canonical URL

    Args:
        result: The CrawlResult object from crawl4ai
        url: The original URL that was crawled

    Returns:
        dict: Metadata dictionary with fields you choose to include

    Example implementation:
        return {
            "url": url,
            "title": result.metadata.get("title", "") if result.metadata else "",
            "description": result.metadata.get("description", ""),
            "word_count": len(result.markdown.raw_markdown.split()) if result.markdown else 0,
            "internal_links": len(result.links.get("internal", [])),
            "external_links": len(result.links.get("external", [])),
            "images": len(result.media.get("images", [])),
        }
    """
    # Placeholder - implement based on your needs
    raise NotImplementedError(
        "Please implement extract_metadata() in c4md.py.\n"
        "See the docstring for guidance on what fields to include."
    )


def save_screenshot(
    result, output_path: Path, quality: int = 85, max_width: int = None
) -> int:
    """
    Decode base64 screenshot, optionally compress, and save.

    Args:
        result: CrawlResult with screenshot data
        output_path: Where to save (extension determines format: .png or .jpg)
        quality: JPEG quality 1-100 (only used for .jpg)
        max_width: Resize to this width if set (maintains aspect ratio)

    Returns:
        File size in bytes
    """
    if not result.screenshot:
        raise click.ClickException("No screenshot data available")

    screenshot_data = base64.b64decode(result.screenshot)
    img = Image.open(BytesIO(screenshot_data))

    # Resize if max_width specified
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

    # Save based on extension
    suffix = output_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        # Convert to RGB (JPEG doesn't support alpha)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=quality, optimize=True)
    else:
        # PNG - use compression
        img.save(output_path, "PNG", optimize=True)

    return output_path.stat().st_size


def save_pdf(result, output_path: Path) -> None:
    """Save PDF bytes to file."""
    if not result.pdf:
        raise click.ClickException("No PDF data available")

    output_path.write_bytes(result.pdf)


def save_mhtml(result, output_path: Path) -> None:
    """Save MHTML string to file."""
    if not result.mhtml:
        raise click.ClickException("No MHTML data available")

    output_path.write_text(result.mhtml, encoding="utf-8")


@click.command()
@click.argument("url")
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["text", "markdown", "screenshot", "pdf", "metadata", "archive"]),
    default="text",
    help="Output mode",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./output",
    help="Output directory",
)
@click.option(
    "-f",
    "--filename",
    default=None,
    help="Custom filename (no extension)",
)
@click.option(
    "--fit",
    is_flag=True,
    help="Use content-filtered markdown (removes boilerplate)",
)
@click.option(
    "--no-headless",
    is_flag=True,
    help="Show browser window",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Verbose progress output",
)
@click.option(
    "--timeout",
    default=30,
    help="Page timeout in seconds",
)
@click.option(
    "--cache",
    type=click.Choice(["bypass", "enabled", "read-only", "write-only"]),
    default="bypass",
    help="Cache mode: bypass (always fresh), enabled (use cache), read-only, write-only",
)
@click.option(
    "--deep",
    is_flag=True,
    help="Deep crawl: follow links 1 level deep",
)
@click.option(
    "--max-pages",
    default=10,
    help="Max pages to crawl in deep mode",
)
@click.option(
    "--sort-by-date",
    is_flag=True,
    help="Sort results by date (newest first) before processing",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit output to N results (applied after sorting)",
)
@click.option(
    "--url-pattern",
    default=None,
    help="Filter deep crawl URLs (glob pattern, e.g., '*garden*' or '*/essays/*')",
)
@click.option(
    "--no-images",
    is_flag=True,
    help="Strip images from markdown (LLM-friendly)",
)
@click.option(
    "--image-quality",
    default=2,
    type=click.IntRange(1, 6),
    help="Image filter threshold 1-6 (higher=stricter)",
)
@click.option(
    "--screenshot-quality",
    default=85,
    type=click.IntRange(1, 100),
    help="JPEG quality for screenshots (use .jpg extension)",
)
@click.option(
    "--screenshot-width",
    default=None,
    type=int,
    help="Max screenshot width (resize if larger)",
)
@click.option(
    "--download-images",
    is_flag=True,
    help="Download and compress images locally (for markdown mode)",
)
@click.option(
    "--embed-images",
    is_flag=True,
    help="Embed images as base64 in markdown (self-contained for LLMs)",
)
@click.option(
    "--image-width",
    default=800,
    type=int,
    help="Max width for downloaded/embedded images",
)
def main(
    url: str,
    mode: str,
    output: str,
    filename: Optional[str],
    fit: bool,
    no_headless: bool,
    verbose: bool,
    timeout: int,
    cache: str,
    deep: bool,
    max_pages: int,
    sort_by_date: bool,
    limit: Optional[int],
    url_pattern: Optional[str],
    no_images: bool,
    image_quality: int,
    screenshot_quality: int,
    screenshot_width: Optional[int],
    download_images: bool,
    embed_images: bool,
    image_width: int,
):
    """
    Crawl a URL and extract content in various formats.

    Examples:

        # Extract clean text
        c4md https://example.com/article

        # Take a screenshot
        c4md https://example.com --mode screenshot

        # Full archive with all formats
        c4md https://example.com --mode archive -v

        # Content-filtered markdown
        c4md https://example.com --mode text --fit

        # Deep crawl - follow links 1 level deep
        c4md https://example.com/blog --mode archive --deep -v

        # LLM-optimized: clean text, no images, compressed screenshots
        c4md https://example.com --mode text --fit --no-images
        c4md https://example.com --mode screenshot --screenshot-width 1200
    """
    output_dir = Path(output)
    slug = filename or url_to_slug(url)

    if verbose:
        mode_str = f"{mode}" + (" (deep)" if deep else "")
        console.print(Panel(f"[bold]c4md - Crawl4AI Markdown Extractor[/bold]\n{url}\nMode: {mode_str}", expand=False))

    # Run the async crawl
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        if verbose:
            desc = f"Deep crawling (max {max_pages} pages)..." if deep else "Crawling..."
            progress.add_task(description=desc, total=None)

        results = asyncio.run(
            crawl_url(
                url=url,
                mode=mode,
                headless=not no_headless,
                timeout=timeout,
                fit=fit,
                verbose=verbose,
                deep=deep,
                max_pages=max_pages,
                no_images=no_images,
                image_quality=image_quality,
                cache_mode_str=cache,
                url_pattern=url_pattern,
            )
        )

    # Normalize to list for consistent handling
    if not isinstance(results, list):
        results = [results]

    if verbose and deep:
        console.print(f"[dim]Crawled {len(results)} pages[/dim]")

    # Sort by date if requested
    if sort_by_date and len(results) > 1:
        if verbose:
            console.print("[dim]Sorting by date (newest first)...[/dim]")
        results = sort_results_by_date(results, descending=True)

        # Show dates in verbose mode
        if verbose:
            for r in results[:5]:  # Show first 5
                date = extract_date_from_result(r)
                date_str = date if date else "no date"
                console.print(f"  [dim]{date_str}: {r.url}[/dim]")
            if len(results) > 5:
                console.print(f"  [dim]... and {len(results) - 5} more[/dim]")

    # Apply limit after sorting
    if limit and len(results) > limit:
        if verbose:
            console.print(f"[dim]Limiting to {limit} results[/dim]")
        results = results[:limit]

    # Show cache status in verbose mode
    if verbose and not deep:
        cache_status = getattr(results[0], "cache_status", None) if results else None
        if cache_status:
            console.print(f"[dim]Cache: {cache_status}[/dim]")

    # Process each result
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        # Generate unique slug for each page in deep mode
        if deep:
            page_slug = url_to_slug(result.url)
            if verbose:
                console.print(f"\n[bold]Page {i+1}/{len(results)}:[/bold] {result.url}")
        else:
            page_slug = slug

        # Process based on mode
        if mode == "text":
            text_path = output_dir / f"{page_slug}_text.md"
            content = extract_text_markdown(result, fit=fit)
            text_path.write_text(content, encoding="utf-8")
            console.print(f"[green]Saved:[/green] {text_path}")

        elif mode == "markdown":
            content, references = extract_full_markdown(result)

            # Embed images as base64 (self-contained markdown for LLMs)
            if embed_images:
                if verbose:
                    console.print("[dim]Downloading and embedding images...[/dim]")
                url_to_base64 = download_images_as_base64(
                    result,
                    quality=screenshot_quality,
                    max_width=image_width,
                )
                if url_to_base64:
                    content = embed_images_in_markdown(content, url_to_base64)
                    if references:
                        references = embed_images_in_markdown(references, url_to_base64)
                    # Calculate embedded size
                    total_b64_size = sum(len(b64) for b64 in url_to_base64.values())
                    if verbose:
                        console.print(
                            f"[green]Embedded:[/green] {len(url_to_base64)} images "
                            f"(~{total_b64_size // 1024}KB base64)"
                        )

            # Optionally download and compress images to files
            elif download_images:
                images_dir = output_dir / f"{page_slug}_images"
                url_to_path = download_and_compress_images(
                    result, images_dir,
                    quality=screenshot_quality,
                    max_width=image_width,
                )
                if url_to_path and verbose:
                    total_size = sum(p.stat().st_size for p in url_to_path.values())
                    console.print(
                        f"[green]Downloaded:[/green] {len(url_to_path)} images "
                        f"({total_size // 1024}KB) to {images_dir}/"
                    )

            md_path = output_dir / f"{page_slug}.md"
            md_path.write_text(content, encoding="utf-8")
            console.print(f"[green]Saved:[/green] {md_path}")

            if references:
                refs_path = output_dir / f"{page_slug}_refs.md"
                refs_path.write_text(references, encoding="utf-8")
                console.print(f"[green]Saved:[/green] {refs_path}")

        elif mode == "screenshot":
            # Use .jpg for compression if width is specified, otherwise PNG
            ext = ".jpg" if screenshot_width else ".png"
            screenshot_path = output_dir / f"{page_slug}{ext}"
            size = save_screenshot(
                result, screenshot_path,
                quality=screenshot_quality,
                max_width=screenshot_width
            )
            size_str = f" ({size // 1024}KB)" if verbose else ""
            console.print(f"[green]Saved:[/green] {screenshot_path}{size_str}")

        elif mode == "pdf":
            pdf_path = output_dir / f"{page_slug}.pdf"
            save_pdf(result, pdf_path)
            console.print(f"[green]Saved:[/green] {pdf_path}")

        elif mode == "metadata":
            meta_path = output_dir / f"{page_slug}_meta.json"
            metadata = extract_metadata(result, result.url)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            console.print(f"[green]Saved:[/green] {meta_path}")

        elif mode == "archive":
            # Create subdirectory for each page's archive
            archive_dir = output_dir / page_slug
            archive_dir.mkdir(parents=True, exist_ok=True)

            files_saved = []

            # Markdown
            content, references = extract_full_markdown(result)
            md_path = archive_dir / "article.md"
            md_path.write_text(content, encoding="utf-8")
            files_saved.append(md_path)

            # Clean text
            text_content = extract_text_markdown(result, fit=fit)
            text_path = archive_dir / "article_text.md"
            text_path.write_text(text_content, encoding="utf-8")
            files_saved.append(text_path)

            # Screenshot
            if result.screenshot:
                ext = ".jpg" if screenshot_width else ".png"
                screenshot_path = archive_dir / f"article{ext}"
                save_screenshot(
                    result, screenshot_path,
                    quality=screenshot_quality,
                    max_width=screenshot_width
                )
                files_saved.append(screenshot_path)

            # PDF
            if result.pdf:
                pdf_path = archive_dir / "article.pdf"
                save_pdf(result, pdf_path)
                files_saved.append(pdf_path)

            # MHTML
            if result.mhtml:
                mhtml_path = archive_dir / "article.mhtml"
                save_mhtml(result, mhtml_path)
                files_saved.append(mhtml_path)

            # Metadata (only if implemented)
            try:
                metadata = extract_metadata(result, result.url)
                meta_path = archive_dir / "metadata.json"
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                files_saved.append(meta_path)
            except NotImplementedError:
                if verbose and i == 0:  # Only warn once
                    console.print("[dim]Skipping metadata (not implemented)[/dim]")

            console.print(f"[green]Archive created:[/green] {archive_dir}/")
            for f in files_saved:
                console.print(f"  [dim]{f.name}[/dim]")

    # Summary for deep crawl
    if deep and verbose:
        console.print(f"\n[bold green]Done![/bold green] Processed {len(results)} pages to {output_dir}/")


if __name__ == "__main__":
    main()
